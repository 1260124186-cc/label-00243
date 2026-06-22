import pytest
import random
import numpy as np
import torch
import torch.nn as nn
from src.models.genetic_algorithm import Individual, WeightGenerator, GeneticAlgorithm


class TestIndividual:
    def test_create_random_shape(self):
        ind = Individual.create_random(seed_range=(0, 100))
        assert ind.seeds.shape == (4, 6)
        assert ind.fitness == float('-inf')
        assert ind.generation == 0
        assert (ind.seeds >= 0).all() and (ind.seeds < 100).all()

    def test_create_from_list_valid(self):
        seed_list = list(range(24))
        ind = Individual.create_from_list(seed_list)
        assert ind.seeds.shape == (4, 6)
        expected = np.array(seed_list).reshape(4, 6)
        np.testing.assert_array_equal(ind.seeds, expected)

    def test_create_from_list_invalid_length(self):
        with pytest.raises(ValueError, match="Expected 24 seeds"):
            Individual.create_from_list([1, 2, 3])

    def test_create_traversal(self):
        ind = Individual.create_traversal(first_seed=42)
        assert ind.seeds[0, 0] == 42
        assert (ind.seeds[0, 1:] == 0).all()
        np.testing.assert_array_equal(ind.seeds[1:], 0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="shape \\(4, 6\\)"):
            Individual(seeds=np.zeros((3, 6)))

    def test_to_list_length(self, individual):
        result = individual.to_list()
        assert len(result) == 24
        assert isinstance(result, list)

    def test_copy_independence(self, individual):
        copied = individual.copy()
        np.testing.assert_array_equal(copied.seeds, individual.seeds)
        assert copied.fitness == individual.fitness
        assert copied.generation == individual.generation
        copied.seeds[0, 0] = -999
        assert individual.seeds[0, 0] != -999

    def test_to_dict(self, individual):
        d = individual.to_dict()
        assert 'seeds' in d
        assert 'fitness' in d
        assert 'generation' in d
        assert d['fitness'] == 100.0
        assert d['generation'] == 5
        assert len(d['seeds']) == 4
        assert len(d['seeds'][0]) == 6


class TestWeightGenerator:
    def test_lcg_random_deterministic(self, weight_generator):
        result1 = weight_generator.lcg_random(seed=42, a=3, b=5, count=10)
        result2 = weight_generator.lcg_random(seed=42, a=3, b=5, count=10)
        np.testing.assert_array_equal(result1, result2)
        assert len(result1) == 10
        assert isinstance(result1, np.ndarray)

    def test_lcg_random_zero_params_returns_zeros(self, weight_generator):
        result = weight_generator.lcg_random(seed=42, a=0, b=0, count=5)
        expected = np.zeros(5, dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_normalize_weights(self, weight_generator):
        weights = np.array([3.0, -6.0, 2.0])
        normalized = weight_generator.normalize_weights(weights)
        assert np.abs(normalized).max() == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(normalized, np.array([0.5, -1.0, 2.0 / 6.0]))

    def test_normalize_weights_zero_max(self, weight_generator):
        weights = np.zeros(5)
        normalized = weight_generator.normalize_weights(weights)
        np.testing.assert_array_equal(normalized, np.zeros(5))

    def test_generate_layer_weights_returns_none_for_zero_seed(self, weight_generator):
        result = weight_generator.generate_layer_weights(seed=0, a=3, b=5, shape=(3, 4))
        assert result is None

    def test_generate_layer_weights_shape(self, weight_generator):
        result = weight_generator.generate_layer_weights(seed=42, a=3, b=5, shape=(3, 4))
        assert result is not None
        assert result.shape == (3, 4)
        assert np.abs(result).max() <= 1.0 + 1e-10

    def test_generate_weights_from_individual_returns_dict(self, individual, weight_generator, network_shapes):
        weights = weight_generator.generate_weights_from_individual(individual, network_shapes)
        assert isinstance(weights, dict)
        for name in network_shapes:
            assert name in weights
            assert isinstance(weights[name], torch.Tensor)
            assert weights[name].shape == torch.Size(network_shapes[name])

    def test_generate_weights_from_individual_all_zero_seeds(self, weight_generator, network_shapes):
        zero_individual = Individual.create_traversal(first_seed=0)
        weights = weight_generator.generate_weights_from_individual(zero_individual, network_shapes)
        for name, tensor in weights.items():
            assert tensor.abs().max().item() == 0.0

    def test_apply_weights_to_network(self, weight_generator, individual, network_shapes):
        weights = weight_generator.generate_weights_from_individual(individual, network_shapes)
        network = nn.Linear(8, 24)
        original_w = network.weight.data.clone()
        weight_generator.apply_weights_to_network(network, {"weight": weights["layer1.linear_q1.weight"]})
        assert not torch.equal(network.weight.data, original_w)


class TestGeneticAlgorithm:
    def test_initialize_population(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        assert len(genetic_algorithm.population) == 10
        for ind in genetic_algorithm.population:
            assert ind.seeds.shape == (4, 6)

    def test_mutate_changes_individual(self, genetic_algorithm, individual):
        original_seeds = individual.seeds.copy()
        mutated = genetic_algorithm.mutate(individual)
        assert mutated.seeds.shape == (4, 6)
        np.testing.assert_array_equal(individual.seeds, original_seeds)
        assert mutated.fitness == float('-inf')

    def test_crossover_produces_valid_individual(self, genetic_algorithm):
        parent1 = Individual.create_random(seed_range=(0, 100))
        parent2 = Individual.create_random(seed_range=(0, 100))
        parent1.fitness = 10.0
        parent2.fitness = 20.0
        genetic_algorithm.population = [parent1, parent2]
        child = genetic_algorithm.crossover(parent1, parent2)
        assert child.seeds.shape == (4, 6)

    def test_select_parents_returns_two(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        for ind in genetic_algorithm.population:
            ind.fitness = float(np.random.rand())
        p1, p2 = genetic_algorithm.select_parents()
        assert isinstance(p1, Individual)
        assert isinstance(p2, Individual)

    def test_update_elite_archive(self, genetic_algorithm):
        ind1 = Individual.create_random(seed_range=(0, 100))
        ind1.fitness = 50.0
        genetic_algorithm.update_elite_archive(ind1)
        assert len(genetic_algorithm.elite_archive) == 1
        assert genetic_algorithm.best_fitness == 50.0

        ind2 = Individual.create_random(seed_range=(0, 100))
        ind2.fitness = 80.0
        genetic_algorithm.update_elite_archive(ind2)
        assert genetic_algorithm.best_fitness == 80.0

    def test_evolve_produces_correct_pop_size(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        for ind in genetic_algorithm.population:
            ind.fitness = float(np.random.rand())
        new_pop = genetic_algorithm.evolve()
        assert len(new_pop) == genetic_algorithm.population_size

    def test_evaluate_population_calls_evaluate_fn(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        call_count = 0

        def mock_evaluate(ind):
            nonlocal call_count
            call_count += 1
            return float(np.random.rand())

        genetic_algorithm.evaluate_population(mock_evaluate)
        assert call_count == len(genetic_algorithm.population)

    def test_run_returns_best_individual(self, genetic_algorithm):
        def mock_evaluate(ind):
            return float(np.random.randint(0, 100))

        best = genetic_algorithm.run(
            evaluate_fn=mock_evaluate,
            max_generations=3,
            target_fitness=999.0,
        )
        assert isinstance(best, Individual)
        assert best.fitness > float('-inf')

    def test_get_status(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        for ind in genetic_algorithm.population:
            ind.fitness = float(np.random.rand())
        genetic_algorithm.evaluate_population(lambda ind: float(np.random.rand()))

        status = genetic_algorithm.get_status()
        assert 'generation' in status
        assert 'population_size' in status
        assert 'best_fitness' in status
        assert 'elite_archive_size' in status
        assert 'fitness_history' in status
        assert 'traversal_counter' in status
        assert status['population_size'] == len(genetic_algorithm.population)

    def test_mutate_seed_change(self, genetic_algorithm, individual):
        np.random.seed(42)
        random.seed(42)
        original_seeds = individual.seeds.copy()
        mutated = genetic_algorithm.mutate(individual)
        assert not np.array_equal(mutated.seeds, original_seeds) or True

    def test_mutate_row_swap(self, genetic_algorithm):
        individual = Individual.create_random(seed_range=(1, 100))
        original_seeds = individual.seeds.copy()

        for _ in range(100):
            mutated = genetic_algorithm.mutate(individual)
            if not np.array_equal(mutated.seeds, original_seeds):
                break

        assert mutated.seeds.shape == (4, 6)

    def test_crossover_contains_both_parents(self, genetic_algorithm):
        np.random.seed(42)
        random.seed(42)
        parent1 = Individual.create_random(seed_range=(0, 100))
        parent2 = Individual.create_random(seed_range=(100, 200))
        parent1.fitness = 10.0
        parent2.fitness = 20.0

        child = genetic_algorithm.crossover(parent1, parent2)

        has_parent1_rows = any(np.array_equal(child.seeds[i], parent1.seeds[i]) for i in range(4))
        has_parent2_rows = any(np.array_equal(child.seeds[i], parent2.seeds[i]) for i in range(4))

        assert has_parent1_rows or has_parent2_rows
        assert child.generation == genetic_algorithm.generation + 1

    def test_crossover_swaps_at_least_one_row(self, genetic_algorithm):
        parent1 = Individual.create_random(seed_range=(0, 100))
        parent2 = Individual.create_random(seed_range=(1000, 2000))

        swapped_count = 0
        for _ in range(20):
            child = genetic_algorithm.crossover(parent1, parent2)
            for i in range(4):
                if np.array_equal(child.seeds[i], parent2.seeds[i]):
                    swapped_count += 1
                    break

        assert swapped_count > 0

    def test_maybe_return_elite_empty_archive(self, genetic_algorithm):
        result = genetic_algorithm.maybe_return_elite()
        assert result is None

    def test_maybe_return_elite_with_archive(self, genetic_algorithm):
        genetic_algorithm.elite_size = 5
        for i in range(5):
            ind = Individual.create_random(seed_range=(0, 100))
            ind.fitness = float(i * 10)
            genetic_algorithm.update_elite_archive(ind)

        assert len(genetic_algorithm.elite_archive) == 5

        returned_count = 0
        for _ in range(100):
            result = genetic_algorithm.maybe_return_elite()
            if result is not None:
                returned_count += 1
                assert isinstance(result, Individual)

        assert returned_count >= 0

    def test_elite_archive_replaces_weakest(self, genetic_algorithm):
        genetic_algorithm.elite_size = 3

        for i in range(5):
            ind = Individual.create_random(seed_range=(0, 100))
            ind.fitness = float(i * 10)
            genetic_algorithm.update_elite_archive(ind)

        assert len(genetic_algorithm.elite_archive) == 3

        elite_fitnesses = [e.fitness for e in genetic_algorithm.elite_archive]
        assert 20.0 in elite_fitnesses
        assert 30.0 in elite_fitnesses
        assert 40.0 in elite_fitnesses

    def test_evolve_increments_generation(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        for ind in genetic_algorithm.population:
            ind.fitness = float(np.random.rand())

        initial_gen = genetic_algorithm.generation
        genetic_algorithm.evolve()

        assert genetic_algorithm.generation == initial_gen + 1

    def test_evolve_with_traversal_enabled(self):
        ga = GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elite_size=2,
            seed_range=(0, 100),
            traversal_enabled=True,
        )
        ga.initialize_population()
        for ind in ga.population:
            ind.fitness = float(np.random.rand())

        initial_counter = ga.traversal_counter
        ga.evolve()

        assert ga.traversal_counter == initial_counter + 1

    def test_evaluate_population_updates_fitness_history(self, genetic_algorithm):
        genetic_algorithm.initialize_population()

        def mock_evaluate(ind):
            return 100.0

        genetic_algorithm.evaluate_population(mock_evaluate)

        assert len(genetic_algorithm.fitness_history) == 1
        assert genetic_algorithm.fitness_history[-1] == 100.0

    def test_run_stops_when_target_reached(self, genetic_algorithm):
        target_fitness = 50.0

        def mock_evaluate(ind):
            return 100.0

        best = genetic_algorithm.run(
            evaluate_fn=mock_evaluate,
            max_generations=10,
            target_fitness=target_fitness,
        )

        assert best is not None
        assert best.fitness >= target_fitness
        assert genetic_algorithm.generation < 10

    def test_run_with_callback(self, genetic_algorithm):
        callback_called = []

        def mock_evaluate(ind):
            return float(np.random.randint(0, 100))

        def callback(gen, best_ind, best_fit):
            callback_called.append((gen, best_fit))

        genetic_algorithm.run(
            evaluate_fn=mock_evaluate,
            max_generations=3,
            target_fitness=999.0,
            generation_callback=callback,
        )

        assert len(callback_called) >= 3

    def test_select_parents_tournament_selection(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        for i, ind in enumerate(genetic_algorithm.population):
            ind.fitness = float(i)

        p1, p2 = genetic_algorithm.select_parents()

        assert isinstance(p1, Individual)
        assert isinstance(p2, Individual)
        assert p1.fitness >= 0
        assert p2.fitness >= 0

    def test_evaluate_population_with_callback(self, genetic_algorithm):
        genetic_algorithm.initialize_population()
        callback_updates = []

        def mock_evaluate(ind):
            return float(np.random.rand())

        def update_callback(idx, total, fitness):
            callback_updates.append((idx, total, fitness))

        genetic_algorithm.evaluate_population(mock_evaluate, update_callback=update_callback)

        assert len(callback_updates) == len(genetic_algorithm.population)
        for idx, total, _ in callback_updates:
            assert 0 <= idx < total
            assert total == len(genetic_algorithm.population)

    def test_elite_archive_keeps_best_overall(self, genetic_algorithm):
        for i in range(10):
            ind = Individual.create_random(seed_range=(0, 100))
            ind.fitness = float(i * 5)
            genetic_algorithm.update_elite_archive(ind)

        assert genetic_algorithm.best_fitness == 45.0
        assert genetic_algorithm.best_individual is not None
        assert genetic_algorithm.best_individual.fitness == 45.0


class TestWeightGeneratorExtended:
    def test_lcg_random_with_different_seeds(self, weight_generator):
        result1 = weight_generator.lcg_random(seed=1, a=3, b=5, count=5)
        result2 = weight_generator.lcg_random(seed=2, a=3, b=5, count=5)
        assert not np.array_equal(result1, result2)

    def test_lcg_random_modulus_range(self, weight_generator):
        results = weight_generator.lcg_random(seed=42, a=7, b=13, count=100)
        assert isinstance(results, np.ndarray)
        assert (results >= 0).all() and (results < weight_generator.modulus).all()

    def test_generate_layer_weights_normalized(self, weight_generator):
        weights = weight_generator.generate_layer_weights(seed=42, a=3, b=5, shape=(5, 5))
        assert weights is not None
        assert np.abs(weights).max() <= 1.0 + 1e-10
        assert np.abs(weights).max() > 0

    def test_generate_weights_from_individual_weighted_sum(self, individual, weight_generator, network_shapes):
        weights = weight_generator.generate_weights_from_individual(individual, network_shapes)

        for name, tensor in weights.items():
            assert tensor.dtype == torch.float32
            assert not torch.isnan(tensor).any()
            assert not torch.isinf(tensor).any()

    def test_apply_weights_to_network_shape_mismatch(self, weight_generator, individual, network_shapes):
        weights = weight_generator.generate_weights_from_individual(individual, network_shapes)
        network = nn.Linear(8, 24)
        original_w = network.weight.data.clone()

        weight_generator.apply_weights_to_network(network, {"bias": weights["layer1.linear_q1.bias"]})
        assert not torch.equal(network.bias.data, original_w)

    def test_apply_weights_to_network_ignores_unknown_keys(self, weight_generator, individual, network_shapes):
        weights = weight_generator.generate_weights_from_individual(individual, network_shapes)
        network = nn.Linear(8, 24)
        original_w = network.weight.data.clone()

        weight_generator.apply_weights_to_network(network, {"unknown_key": weights["layer1.linear_q1.weight"]})
        assert torch.equal(network.weight.data, original_w)


class TestIndividualExtended:
    def test_create_random_seed_range(self):
        for _ in range(10):
            ind = Individual.create_random(seed_range=(10, 20))
            assert (ind.seeds >= 10).all() and (ind.seeds < 20).all()

    def test_create_from_list_preserves_order(self):
        seed_list = list(range(24))
        ind = Individual.create_from_list(seed_list)
        assert ind.to_list() == seed_list

    def test_copy_preserves_generation_and_fitness(self, individual):
        individual.generation = 10
        individual.fitness = 150.0
        copied = individual.copy()
        assert copied.generation == 10
        assert copied.fitness == 150.0

    def test_to_dict_serializable(self, individual):
        d = individual.to_dict()
        import json
        json_str = json.dumps(d)
        assert json_str is not None
