#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

int main(void) {
  srand(time(NULL));
  struct NEAT_Context ctx = NEAT_constructPopulation(&(struct NEAT_Parameters){
    .inputs = 3,
    .outputs = 4,
    .populationSize = 7,
    .allowRecurrent = true,
    .initialSpeciesTarget = 10,
    .initialSpeciationThreshold = 1.5f,
    .improvementDeadline = 15,
  });

  NEAT_printNetwork(&ctx.population[0]);

  return 0;
}
