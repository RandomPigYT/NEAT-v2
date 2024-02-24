#include <stdio.h>
#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

// Returns false if a connection was not added
bool NEAT_createConnection(struct NEAT_Genome *genome, uint32_t to,
                           uint32_t from, struct NEAT_Context *ctx) {
  bool isNovelConnection = true;
  uint64_t innovation = 0;

  for (uint64_t i = 0; i < ctx->history.count; i++) {
    struct NEAT_ConnectionRecord temp = ctx->history.items[i];
    if (temp.generation == ctx->currentGeneration && temp.from == from &&
        temp.to == to) {
      isNovelConnection = false;
      innovation = i;
    }
  }

  if (isNovelConnection) {
    struct NEAT_ConnectionRecord temp = {
      .generation = ctx->currentGeneration,
      .from = from,
      .to = to,
    };

    innovation = ctx->history.count;
    DA_APPEND(&ctx->history, temp);
  } else {
    // Check if the connection already exists in the genome
    for (uint64_t i = 0; i < genome->connections.count; i++) {
      if (genome->connections.items[i].innovation == innovation)
        return false;
    }
  }

  float weight = (((float)rand() / (float)RAND_MAX) *
                  (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                 NEAT_MIN_CON_GEN_RANGE;

  struct NEAT_Connection newConnection = {
    .from = from,
    .to = to,
    .weight = weight,
    .innovation = innovation,
    .enabled = true,
  };

  DA_APPEND(&genome->connections, newConnection);

  // Create new neurons if "from" or "to" do not exist
  bool fromExists = false;
  bool toExists = false;

  for (uint64_t i = 0; i < genome->neurons.count; i++) {
    if (genome->neurons.items[i].id == from) {
      fromExists = true;
    }

    if (genome->neurons.items[i].id == to) {
      toExists = true;
    }
  }

  if (!fromExists) {
    struct NEAT_Neuron newNeuron = {
      .kind = NEAT_NEURON_KIND_HIDDEN,
      .activation = 0.0f,
      .id = from,
    };

    DA_APPEND(&genome->neurons, newNeuron);
  }

  if (!toExists) {
    struct NEAT_Neuron newNeuron = {
      .kind = NEAT_NEURON_KIND_HIDDEN,
      .activation = 0.0f,
      .id = to,
    };

    DA_APPEND(&genome->neurons, newNeuron);
  }

  return true;
}

struct NEAT_Context NEAT_constructPopulation(uint32_t populationSize,
                                             uint32_t targetSpecies,
                                             float speciationThreshold) {
  struct NEAT_Context ctx = {
    .history = { 0 },
    .populationSize = populationSize,
    .population = NULL,
    .targetSpecies = targetSpecies,
    .speciationThreshold = speciationThreshold,
    .currentGeneration = 0,
  };

  ctx.population = malloc(ctx.populationSize * sizeof(struct NEAT_Genome));
  assert(ctx.population != NULL && "Failed to allocate memory for population");

  return ctx;
}

void NEAT_printNetwork() {
  /* ... */
}

int main(void) {
  return 0;
}
