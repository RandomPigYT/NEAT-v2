#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

int64_t NEAT_findNeuron(struct NEAT_Genome *g, uint32_t id) {
  int64_t index = -1;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].id == id)
      index = i;
  }
  return index;
}

uint32_t NEAT_layer(struct NEAT_Genome *g, uint32_t neuronIndex,
                    uint32_t runningLayerCount) {
  if (g->neurons.items[neuronIndex].kind == NEAT_NEURON_KIND_INPUT ||
      g->neurons.items[neuronIndex].kind == NEAT_NEURON_KIND_BIAS) {
    return runningLayerCount;
  }

  DA_CREATE(uint32_t) inConnections = { 0 };

  for (uint32_t i = 0; i < g->connections.count; i++) {
    if (g->connections.items[i].enabled &&
        g->connections.items[i].kind == NEAT_CON_KIND_FORWARD &&
        g->connections.items[i].to == g->neurons.items[neuronIndex].id) {
      DA_APPEND(&inConnections, i);
    }
  }

  if (!inConnections.count) {
    if (runningLayerCount == 0)
      return 1;

    return runningLayerCount;
  }

  uint32_t maxLayer = 0;
  for (uint32_t i = 0; i < inConnections.count; i++) {
    int64_t neuronIndex =
      NEAT_findNeuron(g, g->connections.items[inConnections.items[i]].from);
    assert(neuronIndex >= 0 && "invalid connection");

    uint32_t temp = NEAT_layer(g, neuronIndex, runningLayerCount + 1);
    maxLayer = temp > maxLayer ? temp : maxLayer;
  }

  DA_FREE(&inConnections);
  return maxLayer;
}

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

  NEAT_createConnection(&ctx.population[0], 69, 3, true, &ctx);
  NEAT_createConnection(&ctx.population[0], 420, 69, true, &ctx);
  NEAT_createConnection(&ctx.population[0], 6, 420, false, &ctx);
  NEAT_createConnection(&ctx.population[0], 80085, 420, true, &ctx);
  NEAT_createConnection(&ctx.population[0], 6, 80085, false, &ctx);

  for (uint32_t i = 0; i < ctx.population[0].neurons.count; i++) {
    ctx.population[0].neurons.items[i].layer =
      NEAT_layer(&ctx.population[0], i, 0);
  }

  NEAT_printNetwork(&ctx.population[0]);

  NEAT_cleanup(&ctx);

  return 0;
}
