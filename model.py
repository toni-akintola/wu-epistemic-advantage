import networkx as nx
import numpy as np
import random
from modelpy_abm.main import AgentModel


def constructModel() -> AgentModel:
    def generateInitialData(model: AgentModel):
        initial_data = {
            # bandit arm A is set to a 0.5 success rate in the decision process
            "a_success_rate": 0.5,
            # bandit arm B is a learned parameter for the agent. Initialize randomly
            "b_success_rate": random.uniform(0.01, 0.99),
            # agent evidence learned, will be used to update their belief and others in the network
            "b_evidence": None,
            # population type, 'marginalized' or 'dominant'
            "type": (
                "dominant"
                if random.random() > model.proportion_marginalized
                else "marginalized"
            ),
        }

    def generateTimestepData(model: AgentModel):
        graph = model.get_graph()
        # run the experiments in all the nodes
        for _node, node_data in graph.nodes(data=True):
            # agent pulls the "a" bandit arm
            if node_data["a_success_rate"] > node_data["b_success_rate"]:
                # agent won't have any new evidence gathered for b
                node_data["b_evidence"] = None

            # agent pulls the "b" bandit arm
            else:
                # agent collects evidence
                node_data["b_evidence"] = int(
                    np.random.binomial(
                        model["num_pulls"], model["objective_b"], size=None
                    )
                )

        # define function to calculate posterior belief
        def calculate_posterior(prior_belief: float, num_evidence: float) -> float:
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

        # update the beliefs, based on evidence and neighbors
        for node, node_data in graph.nodes(data=True):
            neighbors = graph.neighbors(node)
            # update belief of "b" on own evidence gathered
            if node_data["b_evidence"] is not None:
                node_data["b_success_rate"] = calculate_posterior(
                    node_data["b_success_rate"], node_data["b_evidence"]
                )

            # update node belief of "b" based on evidence gathered by neighbors
            for neighbor_node in neighbors:
                neighbor_evidence = graph.nodes[neighbor_node]["b_evidence"]
                neighbor_type = graph.nodes[neighbor_node]["type"]

                # update from all neighbors if current node is marginalized
                if (
                    node_data["type"] == "marginalized"
                    and neighbor_evidence is not None
                ):
                    node_data["b_success_rate"] = calculate_posterior(
                        node_data["b_success_rate"], neighbor_evidence
                    )

                # only update from dominant agents if current node is dominant
                elif neighbor_type != "marginalized" and neighbor_evidence is not None:
                    node_data["b_success_rate"] = calculate_posterior(
                        node_data["b_success_rate"], neighbor_evidence
                    )

    model = AgentModel()
    model.update_parameters(
        {
            "num_agents": 3,
            "proportion_marginalized": 1 / 6,
            "num_pulls": 1,
            "objective_b": 0.51,
        }
    )

    model.set_initial_data_function(generateInitialData)
    model.set_timestep_function(generateTimestepData)

    return model
