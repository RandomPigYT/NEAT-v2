#ifndef NEAT_H
#define NEAT_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

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

#define DA_INIT_CAPACITY 1
#define DA_APPEND(arr, item)                                            \
  do {                                                                  \
    if ((arr)->count >= (arr)->capacity) {                              \
      (arr)->capacity = (arr)->capacity == 0 ? DA_INIT_CAPACITY         \
                                             : 2 * (arr)->capacity;     \
      (arr)->items =                                                    \
        realloc((arr)->items, (arr)->capacity * sizeof(*(arr)->items)); \
      assert((arr)->items != NULL && " Failed to allocate memory");     \
    }                                                                   \
    (arr)->items[(arr)->count++] = (item);                              \
  } while (0)

#define DA_FREE(arr)     \
  do {                   \
    free((arr)->items);  \
    (arr)->items = NULL; \
    (arr)->count = 0;    \
    (arr)->capacity = 0; \
  } while (0)

#define DA_DELETE_ITEM(arr, index)                                            \
  do {                                                                        \
    assert((arr)->count != 0 && "Attempted to pop from empty dynamic array"); \
    assert((index) < (arr)->count && "Invalid index");                        \
    if ((index) == (arr)->count - 1) {                                        \
      (arr)->count--;                                                         \
    } else {                                                                  \
      memmove(&((arr)->items[index]), &((arr)->items[index + 1]),             \
              ((arr)->count - index) * sizeof(*(arr)->items));                \
      (arr)->count--;                                                         \
    }                                                                         \
  } while (0)

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

  enum NEAT_ConnectionKind kind;

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

  float fitness;
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

  //uint64_t generation;

  uint64_t innovation;
};

struct NEAT_NetworkArch {
  uint32_t inputs;
  uint32_t outputs;
};

struct NEAT_Context {
  uint64_t globalInnovation;

  // Reinitialised every generation
  DA_CREATE(struct NEAT_ConnectionRecord) history;

  DA_CREATE(struct NEAT_Species) species;

  struct NEAT_NetworkArch arch;

  uint32_t populationSize;
  struct NEAT_Genome *population;

  bool allowRecurrent;

  uint32_t targetSpecies;
  float speciationThreshold;

  uint32_t improvementDeadline;

  uint32_t currentGeneration;
};

struct NEAT_Parameters {
  uint32_t inputs, outputs;
  uint32_t populationSize;

  bool allowRecurrent;

  uint32_t initialSpeciesTarget;
  float initialSpeciationThreshold;

  uint32_t improvementDeadline;
};

// Returns false if a connection was not added
bool NEAT_createConnection(struct NEAT_Genome *genome,
                           enum NEAT_ConnectionKind kind, uint32_t to,
                           uint32_t from, bool createMissingNeurons,
                           struct NEAT_Context *ctx);

struct NEAT_Genome NEAT_constructNetwork(struct NEAT_Context *ctx);

void NEAT_splitConnection(struct NEAT_Genome *g, uint32_t connection,
                          struct NEAT_Context *ctx);

struct NEAT_Context
NEAT_constructPopulation(const struct NEAT_Parameters *parameters);

int64_t NEAT_findNeuron(struct NEAT_Genome *g, uint32_t id);

uint32_t NEAT_layerNeuron(struct NEAT_Genome *g, uint32_t neuronIndex,
                          uint32_t runningLayerCount);

void NEAT_layer(struct NEAT_Context *ctx);

void NEAT_cleanup(struct NEAT_Context *ctx);

void NEAT_printNetwork(struct NEAT_Genome *g);

#ifdef NEAT_H_IMPLEMENTATION

// Returns false if a connection was not added
bool NEAT_createConnection(struct NEAT_Genome *genome,
                           enum NEAT_ConnectionKind kind, uint32_t to,
                           uint32_t from, bool createMissingNeurons,
                           struct NEAT_Context *ctx) {
  // Check if the connection is possible
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

  if ((!fromExists || !toExists) && !createMissingNeurons) {
    return false;
  }

  if (!fromExists) {
    struct NEAT_Neuron n = {
      .id = from,
      .kind = NEAT_NEURON_KIND_HIDDEN,
      .activation = 0.0f,
      .layer = 0,
    };

    DA_APPEND(&genome->neurons, n);
  }

  if (!toExists) {
    struct NEAT_Neuron n = {
      .id = to,
      .kind = NEAT_NEURON_KIND_HIDDEN,
      .activation = 0.0f,
      .layer = 0,
    };

    DA_APPEND(&genome->neurons, n);
  }

  bool isNovelConnection = true;
  uint64_t innovation = 0;

  for (uint64_t i = 0; i < ctx->history.count; i++) {
    struct NEAT_ConnectionRecord temp = ctx->history.items[i];
    if (temp.from == from && temp.to == to) {
      isNovelConnection = false;
      innovation = temp.innovation;
    }
  }

  if (isNovelConnection) {
    struct NEAT_ConnectionRecord temp = {
      .from = from,
      .to = to,
      .innovation = ctx->globalInnovation++,
    };

    innovation = temp.innovation;
    DA_APPEND(&ctx->history, temp);
  } else {
    // Check if the connection already exists in the genome
    for (uint64_t i = 0; i < genome->connections.count; i++) {
      if (genome->connections.items[i].innovation == innovation) {
        return false;
      }
    }
  }

  float weight = (((float)rand() / (float)RAND_MAX) *
                  (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                 NEAT_MIN_CON_GEN_RANGE;

  struct NEAT_Connection newConnection = {
    .from = from,
    .to = to,
    .weight = weight,
    .kind = kind,
    .innovation = innovation,
    .enabled = true,
  };

  DA_APPEND(&genome->connections, newConnection);

  return true;
}

struct NEAT_Genome NEAT_constructNetwork(struct NEAT_Context *ctx) {
  struct NEAT_Genome genome = {
    .connections = { 0 },
    .neurons = { 0 },
    .species = 0,
  };

  struct NEAT_Neuron biasNeuron = {
    .kind = NEAT_NEURON_KIND_BIAS,
    .activation = 1.0f,
    .id = 0,
    .layer = 0,
  };

  DA_APPEND(&genome.neurons, biasNeuron);

  for (uint32_t j = 1; j < ctx->arch.inputs + ctx->arch.outputs; j++) {
    struct NEAT_Neuron n = {
      .kind = NEAT_NEURON_KIND_NONE,
      .activation = 0.0f,
      .id = j,
      .layer = 0,
    };

    if (j < ctx->arch.inputs) {
      n.kind = NEAT_NEURON_KIND_INPUT;

    } else {
      n.kind = NEAT_NEURON_KIND_OUTPUT;
    }

    DA_APPEND(&genome.neurons, n);
  }

  // Fully connect the network
  for (uint32_t j = 0; j < ctx->arch.inputs; j++) {
    for (uint32_t k = ctx->arch.inputs;
         k < ctx->arch.inputs + ctx->arch.outputs; k++) {
      NEAT_createConnection(&genome, NEAT_CON_KIND_FORWARD, k, j, false, ctx);
    }
  }

  return genome;
}

struct NEAT_Context
NEAT_constructPopulation(const struct NEAT_Parameters *parameters) {
  struct NEAT_Context ctx = {
    .history = { 0 },
		.species = { 0 },
    .arch = { 
			.inputs = parameters->inputs + 1, // Extra input for bias
      .outputs = parameters->outputs, 
		},
    .populationSize = parameters->populationSize,
    .population = NULL,
		.allowRecurrent = parameters->allowRecurrent,
    .targetSpecies = parameters->initialSpeciesTarget,
    .speciationThreshold = parameters->initialSpeciationThreshold,
		.improvementDeadline = parameters->improvementDeadline,
    .currentGeneration = 0,
  };

  ctx.population = malloc(ctx.populationSize * sizeof(struct NEAT_Genome));
  assert(ctx.population != NULL && "Failed to allocate memory for population");

  for (uint32_t i = 0; i < ctx.populationSize; i++) {
    ctx.population[i] = NEAT_constructNetwork(&ctx);
  }

  return ctx;
}

void NEAT_splitConnection(struct NEAT_Genome *g, uint32_t connection,
                          struct NEAT_Context *ctx) {
  assert(connection < g->connections.count && "Invalid connection index");
  g->connections.items[connection].enabled = false;

  uint32_t maxNeuronId = 0;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    maxNeuronId = g->neurons.items[i].id > maxNeuronId ? g->neurons.items[i].id
                                                       : maxNeuronId;
  }

  enum NEAT_ConnectionKind kind = g->connections.items[connection].kind;

  NEAT_createConnection(g, kind, maxNeuronId + 1,
                        g->connections.items[connection].from, true, ctx);

  NEAT_createConnection(g, kind, g->connections.items[connection].to,
                        maxNeuronId + 1, true, ctx);
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
    const char *kind;
    switch (g->connections.items[i].kind) {
      case NEAT_CON_KIND_FORWARD:
        kind = "FWD";
        break;
      case NEAT_CON_KIND_RECURRENT:
        kind = "REC";
        break;
      default:
        assert(0 && "Unreachable");
        break;
    }
    printf(
      "innovation: %ld\tfrom: %d\tto: %d\tweight: %f\tkind: %s\tenabled: %s\n",
      g->connections.items[i].innovation, g->connections.items[i].from,
      g->connections.items[i].to, g->connections.items[i].weight, kind,
      g->connections.items[i].enabled ? "true" : "false");
  }
}

int64_t NEAT_findNeuron(struct NEAT_Genome *g, uint32_t id) {
  int64_t index = -1;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].id == id)
      index = i;
  }
  return index;
}

uint32_t NEAT_layerNeuron(struct NEAT_Genome *g, uint32_t neuronIndex,
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

    uint32_t temp = NEAT_layerNeuron(g, neuronIndex, runningLayerCount + 1);
    maxLayer = temp > maxLayer ? temp : maxLayer;
  }

  DA_FREE(&inConnections);
  return maxLayer;
}

void NEAT_layer(struct NEAT_Context *ctx) {
  for (uint32_t i = 0; i < ctx->populationSize; i++) {
    struct NEAT_Genome *g = &ctx->population[i];

    for (uint32_t neuron = 0; neuron < g->neurons.count; neuron++) {
      g->neurons.items[neuron].layer = NEAT_layerNeuron(g, neuron, 0);
    }

    uint32_t maxLayer = 0;
    for (uint32_t neuron = 0; neuron < g->neurons.count; neuron++) {
      if (g->neurons.items[neuron].kind != NEAT_NEURON_KIND_OUTPUT) {
        maxLayer = g->neurons.items[neuron].layer > maxLayer
                     ? g->neurons.items[neuron].layer
                     : maxLayer;
      }
    }

    for (uint32_t neuron = 0; neuron < g->neurons.count; neuron++) {
      if (g->neurons.items[neuron].kind == NEAT_NEURON_KIND_OUTPUT) {
        g->neurons.items[neuron].layer = maxLayer + 1;
      }

      if (g->neurons.items[neuron].layer == 0 &&
          (g->neurons.items[neuron].kind != NEAT_NEURON_KIND_INPUT &&
           g->neurons.items[neuron].kind != NEAT_NEURON_KIND_BIAS)) {
        g->neurons.items[neuron].layer = 1;
      }
    }
  }
}

void NEAT_cleanup(struct NEAT_Context *ctx) {
  for (uint32_t i = 0; i < ctx->populationSize; i++) {
    DA_FREE(&ctx->population[i].connections);
    DA_FREE(&ctx->population[i].neurons);
  }

  DA_FREE(&ctx->history);

  free(ctx->population);
}
#endif // NEAT_H_IMPLEMENTATION

#endif // NEAT_H
