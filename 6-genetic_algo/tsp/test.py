# Perform PMX crossover and ensure no duplicates, with corrected conflict resolution.

def pmx_crossover_fixed(parent1, parent2, crossover_point1, crossover_point2):
    size = len(parent1)

    # Step 1: Initialize the offspring with None, copy the middle segment from the respective parents
    offspring1 = [None] * size
    offspring2 = [None] * size

    # Copy the segment between the crossover points from parent1 to offspring1 and parent2 to offspring2
    offspring1[crossover_point1:crossover_point2] = parent1[crossover_point1:crossover_point2]
    offspring2[crossover_point1:crossover_point2] = parent2[crossover_point1:crossover_point2]

    # Step 2: Mapping the elements that are copied between the crossover points
    mapping1to2 = {parent1[i]: parent2[i] for i in range(crossover_point1, crossover_point2)}
    mapping2to1 = {parent2[i]: parent1[i] for i in range(crossover_point1, crossover_point2)}

    # Step 3: Helper function to resolve duplicates by considering the mapping
    def resolve_conflicts(candidate, mapping):
        while candidate in mapping:
            candidate = mapping[candidate]
        return candidate

    # Step 4: Fill the remaining slots for offspring1 and offspring2, resolving any conflicts
    for i in range(size):
        if i < crossover_point1 or i >= crossover_point2:
            # For offspring1, use elements from parent2
            if parent2[i] not in offspring1:
                offspring1[i] = parent2[i]
            else:
                offspring1[i] = resolve_conflicts(parent2[i], mapping2to1)

            # For offspring2, use elements from parent1
            if parent1[i] not in offspring2:
                offspring2[i] = parent1[i]
            else:
                offspring2[i] = resolve_conflicts(parent1[i], mapping2to1)

    return offspring1, offspring2

parent1 = [4, 2, 3, 1, 0, 5, 6, 8, 9, 7]
parent2 = [7, 9, 1, 2, 3, 5, 0, 6, 8, 4]
crossover_point1 = 1
crossover_point2 = 5
# Perform PMX crossover again on the same parents and crossover points
offspring1_fixed, offspring2_fixed = pmx_crossover_fixed(parent1, parent2, crossover_point1, crossover_point2)
print(offspring1_fixed)
print(offspring2_fixed)