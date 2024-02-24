#ifndef NEAT_H
#define NEAT_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

#define NEAT_MAX_CON_GEN_RANGE 1.0f
#define NEAT_MIN_CON_GEN_RANGE -1.0f

#define TO_STR__(x) #x
#define TO_STR(x) TO_STR__(x)

#define DA_CREATE(type) \
  struct {              \
    type *items;        \
    uint64_t count;     \
    uint64_t capacity;  \
  }

#define DA_INIT_CAPACITY 25

#define DA_APPEND(arr, item)                                            \
  {                                                                     \
    if ((arr)->count >= (arr)->capacity) {                              \
      (arr)->capacity = (arr)->capacity == 0 ? DA_INIT_CAPACITY         \
                                             : 2 * (arr)->capacity;     \
      (arr)->items =                                                    \
        realloc((arr)->items, (arr)->capacity * sizeof(*(arr)->items)); \
      assert((arr)->items != NULL && " Failed to allocate memory");     \
    }                                                                   \
    (arr)->items[(arr)->count++] = (item);                              \
  }

enum NEAT_NeuronKind {

  NEAT_NEURON_KIND_NONE = 0,
  NEAT_NEURON_KIND_INPUT,
  NEAT_NEURON_KIND_OUTPUT,
  NEAT_NEURON_KIND_BIAS,
  NEAT_NEURON_KIND_HIDDEN,

};

enum NEAT_ActivationFunction {

  NEAT_ACTIVATION_NONE = 0,

  NEAT_ACTIVATION_SIGMOID,
  NEAT_ACTIVATION_TANH,
  NEAT_ACTIVATION_RELU,
  NEAT_ACTIVATION_LEAKY_RELU,

  /* ... */

};

enum NEAT_ConnectionKind {

  NEAT_CON_KIND_NONE = 0,

  NEAT_CON_KIND_FORWARD,
  NEAT_CON_KIND_RECURRENT,

};

struct NEAT_Connection {
  uint32_t from;
  uint32_t to;

  float weight;

  uint64_t innovation;

  bool enabled;
};

struct NEAT_Neuron {
  // id 0 is always the bias neuron
  uint32_t id;

  enum NEAT_NeuronKind kind;
  float activation;

  uint32_t layer;
};

struct NEAT_Genome {
  DA_CREATE(struct NEAT_Connection) connections;
  DA_CREATE(struct NEAT_Neuron) neurons;

  uint32_t species;
};

struct NEAT_Species {
  DA_CREATE(struct NEAT_Genome *) genomes;

  float currentFitness;

  float bestFitness;
  uint32_t gensSinceImproved;
};

struct NEAT_ConnectionRecord {
  uint32_t to;
  uint32_t from;

  uint64_t generation;
};

struct NEAT_NetworkArch {
  uint32_t inputs;
  uint32_t outputs;
};

struct NEAT_Context {
  // The index of the record determines the innovation
  // DO NOT CHANGE ITS ORDER!
  DA_CREATE(struct NEAT_ConnectionRecord) history;

  struct NEAT_NetworkArch arch;

  uint32_t populationSize;
  struct NEAT_Genome *population;

  uint32_t targetSpecies;
  float speciationThreshold;

  uint32_t currentGeneration;
};

// Returns false if a connection was not added
bool NEAT_createConnection(struct NEAT_Genome *genome, uint32_t to,
                           uint32_t from, struct NEAT_Context *ctx);

struct NEAT_Context NEAT_constructPopulation(uint32_t inputs, uint32_t outputs,
                                             uint32_t populationSize,
                                             uint32_t targetSpecies,
                                             float speciationThreshold);

void NEAT_printNetwork(struct NEAT_Genome *g);

#ifdef NEAT_H_IMPLEMENTATION

// Library stuff goes here

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

#endif // NEAT_H_IMPLEMENTATION

#endif // NEAT_H
