import networkx as nx
import numpy as np
import random
from emergent.main import AgentModel


def generateInitialData(model: AgentModel):
    if model["model_variation"] == "base":
        return {
            "a_success_rate": model["objective_a"],
            "b_superior": random.uniform(0.01, 0.99),
            "b_evidence": None,
            "type": (
                "dominant"
                if random.random() > model["proportion_marginalized"]
                else "marginalized"
            ),
        }
    elif model["model_variation"] == "homophily":
        return {
            "a_superior": model["objective_a"],
            "b_superior": random.uniform(0.01, 0.99),
            "b_evidence": None,
            "type": (
                "dominant"
                if random.random() > model["proportion_marginalized"]
                else "marginalized"
            ),
        }
    else:  # devaluation
        graph = model.get_graph()
        num_dominants = sum(
            (
                1
                for node, node_data in graph.nodes(data=True)
                if (node_data.get("type") and node_data["type"] == "dominant")
            )
        )
        return {
            "a_superior": model["objective_a"],
            "b_superior": random.uniform(0.01, 0.99),
            "b_evidence": None,
            "type": (
                "marginalized"
                if ((len(graph.nodes) - num_dominants)) / len(graph.nodes)
                < model["proportion_marginalized"]
                else "dominant"
            ),
        }


def generateTimestepData(model: AgentModel):

    def calculate_posterior_base(prior_belief: float, num_evidence: float) -> float:
        # Calculate likelihood, will be either the success rate
        pEH_likelihood = (model["objective_b"] ** num_evidence) * (
            (1 - model["objective_b"]) ** (model["num_pulls"] - num_evidence)
        )

        # Calculate normalization constant
        pE_evidence = (pEH_likelihood * prior_belief) + (
            (1 - model["objective_b"]) ** num_evidence
        ) * (model["objective_b"] ** (model["num_pulls"] - num_evidence)) * (
            1 - prior_belief
        )

        # Calculate posterior belief using Bayes' theorem
        posterior = (pEH_likelihood * prior_belief) / pE_evidence

        return posterior

    def calculate_posterior_homophily(
        prior_belief: float, num_evidence: float, devalue=False
    ) -> float:
        # Calculate likelihood
        pEH_likelihood = (model["objective_b"] ** num_evidence) * (
            (1 - model["objective_b"]) ** (model["num_pulls"] - num_evidence)
        )

        # Calculate normalization constant
        if devalue:
            pE_evidence = 1 - model["degree_devaluation"] * (1 - prior_belief)
        else:
            pE_evidence = (pEH_likelihood * prior_belief) + (
                (1 - model["objective_b"]) ** num_evidence
            ) * (model["objective_b"] ** (model["num_pulls"] - num_evidence)) * (
                1 - prior_belief
            )

        # Calculate posterior belief using Bayes' theorem
        posterior = (pEH_likelihood * prior_belief) / pE_evidence

        return posterior

    def calculate_posterior_devaluation(
        prior_belief: float, num_evidence: float, devalue=False
    ) -> float:
        # Calculate likelihood P(E|H)
        pEH_likelihood = (model["objective_b"] ** num_evidence) * (
            (1 - model["objective_b"]) ** (model["num_pulls"] - num_evidence)
        )

        # Calculate P(E|¬H)
        pEnH_likelihood = ((1 - model["objective_b"]) ** num_evidence) * (
            model["objective_b"] ** (model["num_pulls"] - num_evidence)
        )

        if devalue:
            # Calculate P_f(E) using the devaluation formula
            pf_E = 1 - model["degree_devaluation"] * (1 - prior_belief)
        else:
            pf_E = 1  # Fully trust the evidence

        # Calculate P_f(¬E)
        pf_notE = 1 - pf_E

        # Calculate posterior using Jeffrey conditionalization
        posterior = (
            pEH_likelihood * prior_belief * pf_E
            + pEnH_likelihood * prior_belief * pf_notE
        ) / (
            (pEH_likelihood * prior_belief + pEnH_likelihood * (1 - prior_belief))
            * pf_E
            + (pEnH_likelihood * prior_belief + pEH_likelihood * (1 - prior_belief))
            * pf_notE
        )

        return posterior

    graph = model.get_graph()

    # Run the experiments in all the nodes
    for _node, node_data in graph.nodes(data=True):
        # Determine which arm to pull based on model variation
        if model["model_variation"] == "base":
            if node_data["a_success_rate"] > node_data["b_superior"]:
                node_data["b_evidence"] = None
            else:
                node_data["b_evidence"] = int(
                    np.random.binomial(
                        model["num_pulls"], model["objective_b"], size=None
                    )
                )
        else:  # homophily and devaluation
            if node_data["a_superior"] > node_data["b_superior"]:
                node_data["b_evidence"] = None
            else:
                node_data["b_evidence"] = int(
                    np.random.binomial(
                        model["num_pulls"], model["objective_b"], size=None
                    )
                )

    # Update beliefs based on evidence and neighbors
    for node, node_data in graph.nodes(data=True):
        neighbors = graph.neighbors(node)

        # Update belief based on own evidence
        if node_data["b_evidence"] is not None:
            if model["model_variation"] == "base":
                node_data["b_superior"] = calculate_posterior_base(
                    node_data["b_superior"], node_data["b_evidence"]
                )
            else:  # homophily and devaluation
                node_data["b_superior"] = calculate_posterior_homophily(
                    node_data["b_superior"], node_data["b_evidence"]
                )

        # Update belief based on neighbor evidence
        for neighbor_node in neighbors:
            neighbor_evidence = graph.nodes[neighbor_node]["b_evidence"]
            neighbor_type = graph.nodes[neighbor_node]["type"]

            if neighbor_evidence is not None:
                if model["model_variation"] == "base":
                    # Base model: marginalized agents update from all, dominant only from dominant
                    if (
                        node_data["type"] == "marginalized"
                        or neighbor_type != "marginalized"
                    ):
                        node_data["b_superior"] = calculate_posterior_base(
                            node_data["b_superior"], neighbor_evidence
                        )
                elif model["model_variation"] == "homophily":
                    # homophily model: marginalized agents update from all, dominant only from dominant
                    if (
                        node_data["type"] == "marginalized"
                        or neighbor_type != "marginalized"
                    ):
                        node_data["b_superior"] = calculate_posterior_homophily(
                            node_data["b_superior"], neighbor_evidence
                        )
                else:  # devaluation
                    # devaluation model: marginalized agents update from all, dominant agents devalue evidence
                    if node_data["type"] == "marginalized":
                        node_data["b_superior"] = calculate_posterior_devaluation(
                            node_data["b_superior"], neighbor_evidence
                        )
                    else:
                        node_data["b_superior"] = calculate_posterior_devaluation(
                            node_data["b_superior"], neighbor_evidence, devalue=True
                        )

    model.set_graph(graph)


def constructModel() -> AgentModel:
    model = AgentModel()

    # Define parameters based on model variation
    base_params = {
        "num_nodes": 20,
        "proportion_marginalized": round(float(1 / 6), 2),
        "num_pulls": 1,
        "objective_b": 0.51,
        "objective_a": 0.5,
    }

    homophily_params = {
        "num_nodes": 20,
        "proportion_marginalized": float(1 / 6),
        "num_pulls": 1,
        "objective_b": 0.51,
        "objective_a": 0.5,
        "p_ingroup": 0.7,
        "p_outgroup": 0.3,
        "degree_devaluation": 0.2,
    }

    # Set model variation parameter (default to "base")
    model["model_variation"] = "base"

    # Update parameters based on model variation
    if model["model_variation"] == "base":
        model.update_parameters(base_params)
    else:
        model.update_parameters(homophily_params)

    model["variations"] = ["base", "homophily", "devaluation"]
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model
