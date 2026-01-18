import networkx as nx
import numpy as np
import random
from emergent.main import AgentModel


def generateHomophilicGraph(model: AgentModel) -> None:
    """
    Regenerate the graph structure to be homophilic, where agents connect with 
    higher probability to agents of the same type (ingroup) than to agents of 
    different types (outgroup).
    
    This function modifies the existing graph in-place by removing all edges
    and regenerating them based on homophily probabilities.
    
    Args:
        model: The AgentModel instance
    """
    graph = model.get_graph()
    p_ingroup = model.get("p_ingroup", 0.7)
    p_outgroup = model.get("p_outgroup", 0.3)
    
    # Remove all existing edges
    graph.remove_edges_from(list(graph.edges()))
    
    # Generate new edges based on homophily
    nodes = list(graph.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            
            # Get agent types
            i_type = graph.nodes[node_i].get("type")
            j_type = graph.nodes[node_j].get("type")
            
            # Determine connection probability based on whether types match
            if i_type == j_type:
                # Same type: use ingroup probability
                p_connect = p_ingroup
            else:
                # Different types: use outgroup probability
                p_connect = p_outgroup
            
            # Add edge with the determined probability
            if random.random() < p_connect:
                graph.add_edge(node_i, node_j)
    
    # Update the model with the modified graph
    model.set_graph(graph)


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
    
    # Regenerate graph structure for homophily variation if needed
    # This should happen once after initial data is set up
    if model["model_variation"] == "homophily":
        # Check if homophilic graph has been initialized
        # Use a flag stored in the model to track this
        if not model.get("_homophilic_graph_initialized", False):
            generateHomophilicGraph(model)
            model["_homophilic_graph_initialized"] = True
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
        "graph_type": "custom",
        "objective_a": 0.5,
        "p_ingroup": 0.7,
        "p_outgroup": 0.3,
        "degree_devaluation": 0.2,
    }

    devaluation_params = {
        "num_nodes": 20,
        "proportion_marginalized": float(1 / 6),
        "num_pulls": 1,
        "objective_b": 0.51,
        "objective_a": 0.5,
        "degree_devaluation": 0.2,
    }

    # Set model variation parameter (default to "base")
    model["model_variation"] = "base"

    # Update parameters based on model variation
    if model["model_variation"] == "base":
        model.update_parameters(base_params)
    elif model["model_variation"] == "homophily":
        model.update_parameters(homophily_params)
    else:  # devaluation
        model.update_parameters(devaluation_params)
    
    model["variations"] = ["base", "homophily", "devaluation"]
    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model
