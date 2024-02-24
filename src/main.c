#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

int main(void) {
  srand(time(NULL));
  struct NEAT_Context ctx = NEAT_constructPopulation(3, 4, 3, 5, 1.5);

  for (uint32_t i = 0; i < 1; i++) {
    NEAT_printNetwork(&ctx.population[i]);
  }

  return 0;
}
