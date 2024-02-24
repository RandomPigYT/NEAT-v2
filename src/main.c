#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

struct NEAT_Context NEAT_constructPopulation(uint32_t inputs, uint32_t outputs,
                                             uint32_t populationSize,
                                             uint32_t targetSpecies,
                                             float speciationThreshold) {
  struct NEAT_Context ctx = {
    .history = { 0 },
    .arch = { .inputs = inputs + 1, .outputs = outputs },
    .populationSize = populationSize,
    .population = NULL,
    .targetSpecies = targetSpecies,
    .speciationThreshold = speciationThreshold,
    .currentGeneration = 0,
  };

  ctx.population = malloc(ctx.populationSize * sizeof(struct NEAT_Genome));
  assert(ctx.population != NULL && "Failed to allocate memory for population");

  for (uint32_t i = 0; i < ctx.populationSize; i++) {
    // Add neurons
    struct NEAT_Neuron biasNeuron = {
      .kind = NEAT_NEURON_KIND_BIAS,
      .activation = 1.0f,
      .id = 0,
      .layer = 0,
    };

    DA_APPEND(&ctx.population[i].neurons, biasNeuron);

    for (uint32_t j = 1; j < ctx.arch.inputs; j++) {
      struct NEAT_Neuron n = {
        .kind = NEAT_NEURON_KIND_NONE,
        .activation = 0.0f,
        .id = j,
        .layer = 0,
      };

      if (j < ctx.arch.inputs) {
        n.kind = NEAT_NEURON_KIND_INPUT;

      } else {
        n.kind = NEAT_NEURON_KIND_OUTPUT;
      }

      DA_APPEND(&ctx.population[i].neurons, n)
    }

    // Fully connect the network
    for (uint32_t j = 0; j < ctx.arch.inputs; j++) {
      for (uint32_t k = ctx.arch.inputs; k < ctx.arch.inputs + ctx.arch.outputs;
           k++) {
        NEAT_createConnection(&ctx.population[i], k, j, &ctx);
      }
    }
  }

  return ctx;
}

void NEAT_printNetwork(struct NEAT_Genome *g) {
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    const char *kind;
    switch (g->neurons.items[i].kind) {
      case NEAT_NEURON_KIND_INPUT:
        kind = "input";
        break;
      case NEAT_NEURON_KIND_OUTPUT:
        kind = "output";
        break;
      case NEAT_NEURON_KIND_BIAS:
        kind = "bias";
        break;
      case NEAT_NEURON_KIND_HIDDEN:
        kind = "hidden";
        break;

      default:
        assert(0 && "Unreachable");
        break;
    }

    printf("id: %d\tkind: %s\tlayer: %d\tactivation: %f\n",
           g->neurons.items[i].id, kind, g->neurons.items[i].layer,
           g->neurons.items[i].activation);
  }

  for (uint32_t i = 0; i < g->connections.count; i++) {
    printf("innovation: %ld\tfrom: %d\tto: %d\tweight: %f\tenabled: %s\n",
           g->connections.items[i].innovation, g->connections.items[i].from,
           g->connections.items[i].to, g->connections.items[i].weight,
           g->connections.items[i].enabled ? "true" : "false");
  }
}

int main(void) {
  srand(time(NULL));
  struct NEAT_Context ctx = NEAT_constructPopulation(3, 4, 3, 5, 1.5);

  for (uint32_t i = 0; i < 1; i++) {
    NEAT_printNetwork(&ctx.population[i]);
  }

  return 0;
}
