#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <raylib.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

// This code is actual garbage
void drawNetwork(struct NEAT_Genome *g, uint32_t x, uint32_t y, uint32_t w,
                 uint32_t h) {
  float radius = 0.01f * w;

  uint32_t numLayers = 0;
  for (uint32_t i = 0; i < g->neurons.count; i++) {
    if (g->neurons.items[i].kind == NEAT_NEURON_KIND_OUTPUT) {
      numLayers = g->neurons.items[i].layer + 1;
      break;
    }
  }

  assert(numLayers > 1);

  uint32_t hpad = w / (numLayers - 1);
  for (uint32_t i = 0; i < numLayers; i++) {
    uint32_t vpad = 0;
    DA_CREATE(uint32_t) neuronsInLayer = { 0 };
    for (uint32_t j = 0; j < g->neurons.count; j++) {
      if (g->neurons.items[j].layer == i) {
        DA_APPEND(&neuronsInLayer, j);
      }
    }

    vpad = h / neuronsInLayer.count;

    for (uint32_t j = 0; j < neuronsInLayer.count; j++) {
      Color colour = WHITE;
      uint32_t cx1 = x + radius + i * hpad;
      uint32_t cy1 = (y + radius) +
                     ((h - (neuronsInLayer.count - 1) * vpad) / 2.0f) +
                     j * vpad;

      DA_CREATE(uint32_t) connections = { 0 };

      for (uint32_t k = 0; k < g->connections.count; k++) {
        if (g->connections.items[k].from ==
              g->neurons.items[neuronsInLayer.items[j]].id &&
            g->connections.items[k].enabled) {
          DA_APPEND(&connections, k);
        }
      }

      DA_CREATE(uint32_t) toNeurons = { 0 };

      for (uint32_t k = 0; k < connections.count; k++) {
        DA_APPEND(
          &toNeurons,
          NEAT_findNeuron(g, g->connections.items[connections.items[k]].to));
      }

      for (uint32_t k = 0; k < toNeurons.count; k++) {
        uint32_t indexInLayer = 0;
        uint32_t neuronsInLayer2 = 0;
        for (uint32_t l = 0; l < g->neurons.count; l++) {
          if (l == toNeurons.items[k]) {
            break;
          }

          if (g->neurons.items[l].layer ==
              g->neurons.items[toNeurons.items[k]].layer) {
            indexInLayer++;
          }
        }

        for (uint32_t l = 0; l < g->neurons.count; l++) {
          if (g->neurons.items[l].layer ==
              g->neurons.items[toNeurons.items[k]].layer) {
            neuronsInLayer2++;
          }
        }
        uint32_t vpad2 = h / neuronsInLayer2;

        uint32_t cx2 =
          x + radius + g->neurons.items[toNeurons.items[k]].layer * hpad;
        uint32_t cy2 = (y + radius) +
                       ((h - (neuronsInLayer2 - 1) * vpad2) / 2.0f) +
                       indexInLayer * vpad2;

        float weight = g->connections.items[connections.items[k]].weight;
        float thickness = 0.005f * h;

        colour = weight >= 0 ? GREEN : RED;

        //uint8_t alpha = fabs(weight) > 1.0f ? 255 : weight * 255;

        uint8_t alpha = tanh(fabs(weight)) * 255;
        // colour = ColorAlphaBlend(
        //   colour, CLITERAL(Color){ .a = alpha, .r = 25, .g = 25, .b = 25 },
        //   WHITE);

        colour = ColorAlphaBlend(
          colour, CLITERAL(Color){ .a = alpha, .r = 35, .g = 35, .b = 35 },
          WHITE);

        DrawLineEx(CLITERAL(Vector2){ cx1, cy1 }, CLITERAL(Vector2){ cx2, cy2 },
                   thickness, colour);

        //DrawLine(cx1, cy1, cx2, cy2, colour);
      }

      DA_FREE(&toNeurons);

      DA_FREE(&connections);

      switch (g->neurons.items[neuronsInLayer.items[j]].kind) {
        case NEAT_NEURON_KIND_INPUT:
          colour = CLITERAL(Color) RED;
          break;
        case NEAT_NEURON_KIND_BIAS:
          colour = RAYWHITE;
          break;
        case NEAT_NEURON_KIND_HIDDEN:
          colour = GREEN;
          break;
        case NEAT_NEURON_KIND_OUTPUT:
          colour = BLUE;
          break;
        default:
          assert(0 && "Unreachable");
          break;
      }

      DrawCircle(cx1, cy1, radius, colour);
    }

    DA_FREE(&neuronsInLayer);
  }
}

void NEAT_numNeuronConnections(const struct NEAT_Genome *g, uint32_t neuron,
                               uint32_t *incoming, uint32_t *outgoing) {
  assert(neuron < g->neurons.count && "Invalid neuron index");

  *incoming = 0;
  *outgoing = 0;

  uint32_t neuronId = g->neurons.items[neuron].id;
  for (uint32_t i = 0; i < g->connections.count; i++) {
    if (g->connections.items[i].to == neuronId) {
      (*incoming)++;
    }

    if (g->connections.items[i].from == neuronId) {
      (*outgoing)++;
    }
  }
}

void NEAT_mutate(struct NEAT_Genome *g, enum NEAT_Phase phase,
                 struct NEAT_Context *ctx) {
  // Types of mutations:
  // -> nudging the weight of a connection
  // -> randomizing the wieght of a connection
  // -> adding a connection
  // -> adding a neuron
  // -> removing a connection
  // -> removing a neuron

  float conToggleProb = ctx->conToggleProbability;
  float nudgeProb = ctx->weightNudgeProbability;
  float randProb = ctx->weightRandomizeProbability;
  float conStructureModProb = phase == NEAT_COMPLEXIFY
                                ? ctx->connectionAddProbability
                                : ctx->connectionDeleteProbability;
  float neuronStructureModProb = phase == NEAT_COMPLEXIFY
                                   ? ctx->neuronAddProbability
                                   : ctx->neuronDeleteProbability;

  if (g->connections.count == 0) {
    if (phase == NEAT_PRUNE) {
      return;
    }

    conStructureModProb = 1.0f;

    neuronStructureModProb = 0.0f;
    randProb = 0.0f;
    nudgeProb = 0.0f;
    conToggleProb = 0.0f;
  }

  // Check if there are no hidden neurons during the purning phase
  if (g->neurons.count == ctx->arch.inputs + ctx->arch.outputs &&
      phase == NEAT_PRUNE) {
    //nudgeProb += neuronStructureModProb / 4;
    //randProb += neuronStructureModProb / 4;
    //conStructureModProb += neuronStructureModProb / 4;
    //conToggleProb += neuronStructureModProb / 4;

    neuronStructureModProb = 0.0f;
  }

  DA_CREATE(uint32_t) removableNeurons = { 0 };

  // Check if there are no neurons with at least one end that is only connected to one connection
  if (phase == NEAT_PRUNE) {
    for (uint32_t i = 0; i < g->neurons.count; i++) {
      if (g->neurons.items[i].kind != NEAT_NEURON_KIND_HIDDEN) {
        continue;
      }

      uint32_t numIncoming = 0;
      uint32_t numOutgoing = 0;

      NEAT_numNeuronConnections(g, i, &numIncoming, &numOutgoing);

      if (numIncoming == 1 || numOutgoing == 1 || numIncoming == 0 ||
          numOutgoing == 0) {
        DA_APPEND(&removableNeurons, i);
      }
    }

    if (removableNeurons.count == 0) {
      //nudgeProb += neuronStructureModProb / 4;
      //randProb += neuronStructureModProb / 4;
      //conStructureModProb += neuronStructureModProb / 4;
      //conToggleProb += neuronStructureModProb / 4;

      neuronStructureModProb = 0.0f;
    }
  }

  float runningSum = 0.0f;
  float random = (float)rand() / (float)RAND_MAX;

  runningSum += nudgeProb;
  if (random < runningSum) {
    assert(g->connections.count > 0);

    uint32_t weightToNudge = rand() % g->connections.count;

    float amount = (((float)rand() / (float)RAND_MAX) *
                    (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                   NEAT_MIN_CON_GEN_RANGE;

    g->connections.items[weightToNudge].weight += amount;

    goto Exit;
  }

  runningSum += randProb;
  if (random < runningSum) {
    assert(g->connections.count > 0);

    uint32_t weightToRandomize = rand() % g->connections.count;

    float newWeight = (((float)rand() / (float)RAND_MAX) *
                       (NEAT_MAX_CON_GEN_RANGE - NEAT_MIN_CON_GEN_RANGE)) +
                      NEAT_MIN_CON_GEN_RANGE;

    g->connections.items[weightToRandomize].weight = newWeight;

    goto Exit;
  }

  runningSum += conToggleProb;
  if (random < runningSum) {
    assert(g->connections.count > 0);

    uint32_t weightToToggle = rand() % g->connections.count;

    g->connections.items[weightToToggle].enabled =
      !g->connections.items[weightToToggle].enabled;

    goto Exit;
  }

  runningSum += conStructureModProb;
  if (random < runningSum) {
    if (phase == NEAT_COMPLEXIFY) {
      uint32_t to = 0;
      uint32_t from = 0;

      enum NEAT_ConnectionKind kind = NEAT_CON_KIND_NONE;

      if (ctx->allowRecurrent) {
        to = rand() % g->neurons.count;
        from = rand() % g->neurons.count;

        kind = g->neurons.items[to].layer > g->neurons.items[from].layer
                 ? NEAT_CON_KIND_FORWARD
                 : NEAT_CON_KIND_RECURRENT;
      }

      else {
        const int maxIters = 100;
        int iter = 0;

        kind = NEAT_CON_KIND_FORWARD;

        do {
          to = rand() % g->neurons.count;
          from = rand() % g->neurons.count;

        } while (g->neurons.items[to].layer <= g->neurons.items[from].layer &&
                 iter++ < maxIters);

        if (iter == maxIters) {
          goto Exit;
        }
      }

      NEAT_createConnection(g, kind, to, from, false, ctx);

      goto Exit;

    } else {
      assert(g->connections.count > 0);

      uint32_t conToRemove = rand() % g->connections.count;

      DA_DELETE_ITEM(&g->connections, conToRemove);
      goto Exit;
    }
  }

  runningSum += neuronStructureModProb;
  if (random < runningSum) {
    if (phase == NEAT_COMPLEXIFY) {
      assert(g->connections.count != 0);

      DA_CREATE(uint32_t) enabledCons = { 0 };
      for (uint32_t i = 0; i < g->connections.count; i++) {
        if (g->connections.items[i].enabled) {
          DA_APPEND(&enabledCons, i);
        }
      }

      if (enabledCons.count == 0) {
        DA_FREE(&enabledCons);
        goto Exit;
      }

      uint32_t conToSplit = enabledCons.items[rand() % enabledCons.count];

      g->connections.items[conToSplit].enabled = false;
      enum NEAT_ConnectionKind kind = g->connections.items[conToSplit].kind;

      // Implicitly creates the new node
      NEAT_createConnection(g, kind, g->nextNeuronId,
                            g->connections.items[conToSplit].from, true, ctx);

      NEAT_createConnection(g, kind, g->connections.items[conToSplit].to,
                            g->nextNeuronId, false, ctx);

      g->nextNeuronId++;

      DA_FREE(&enabledCons);
      goto Exit;

    } else {
      assert(removableNeurons.count != 0);

      uint32_t neuronToRemove =
        removableNeurons.items[rand() % removableNeurons.count];

      uint32_t neuronId = g->neurons.items[neuronToRemove].id;

      uint32_t numIncoming, numOutgoing;
      NEAT_numNeuronConnections(g, neuronToRemove, &numIncoming, &numOutgoing);

      if (numIncoming == 0 || numOutgoing == 0) {
        DA_CREATE(uint32_t) conIdsToRemove = { 0 }; // Innovations, not indices

        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].to == neuronId ||
              g->connections.items[i].from == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
          }
        }

        while (conIdsToRemove.count) {
          for (uint32_t i = 0; i < g->connections.count; i++) {
            if (g->connections.items[i].innovation ==
                conIdsToRemove.items[conIdsToRemove.count - 1]) {
              DA_DELETE_ITEM(&g->connections, i);
              DA_DELETE_ITEM(&conIdsToRemove, conIdsToRemove.count - 1);

              break;
            }
          }
        }

        DA_FREE(&conIdsToRemove);

      } else if (numIncoming == 1) {
        DA_CREATE(uint32_t) conIdsToRemove = { 0 }; // Innovations, not indices
        DA_CREATE(uint32_t) outgoingNeuronIds = { 0 };

        uint32_t incomingNeuronId = 0;
        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].to == neuronId) {
            incomingNeuronId = g->connections.items[i].from;
            break;
          }
        }

        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].to == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);

          } else if (g->connections.items[i].from == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
            DA_APPEND(&outgoingNeuronIds, g->connections.items[i].to);
          }
        }

        while (conIdsToRemove.count) {
          for (uint32_t i = 0; i < g->connections.count; i++) {
            if (g->connections.items[i].innovation ==
                conIdsToRemove.items[conIdsToRemove.count - 1]) {
              DA_DELETE_ITEM(&g->connections, i);
              DA_DELETE_ITEM(&conIdsToRemove, conIdsToRemove.count - 1);

              break;
            }
          }
        }

        for (uint32_t i = 0; i < outgoingNeuronIds.count; i++) {
          uint32_t incoming = NEAT_findNeuron(g, incomingNeuronId);
          uint32_t outgoing = NEAT_findNeuron(g, outgoingNeuronIds.items[i]);

          enum NEAT_ConnectionKind kind = g->neurons.items[outgoing].layer >
                                              g->neurons.items[incoming].layer
                                            ? NEAT_CON_KIND_FORWARD
                                            : NEAT_CON_KIND_RECURRENT;

          NEAT_createConnection(g, kind, outgoingNeuronIds.items[i],
                                incomingNeuronId, false, ctx);
        }

        DA_FREE(&conIdsToRemove);
        DA_FREE(&outgoingNeuronIds);

      } else if (numOutgoing == 1) {
        DA_CREATE(uint32_t) conIdsToRemove = { 0 }; // Innovations, not indices
        DA_CREATE(uint32_t) incomingNeuronIds = { 0 };

        uint32_t outgoingNeuronId = 0;
        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].from == neuronId) {
            outgoingNeuronId = g->connections.items[i].to;
            break;
          }
        }

        for (uint32_t i = 0; i < g->connections.count; i++) {
          if (g->connections.items[i].from == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);

          } else if (g->connections.items[i].to == neuronId) {
            DA_APPEND(&conIdsToRemove, g->connections.items[i].innovation);
            DA_APPEND(&incomingNeuronIds, g->connections.items[i].from);
          }
        }

        while (conIdsToRemove.count) {
          for (uint32_t i = 0; i < g->connections.count; i++) {
            if (g->connections.items[i].innovation ==
                conIdsToRemove.items[conIdsToRemove.count - 1]) {
              DA_DELETE_ITEM(&g->connections, i);
              DA_DELETE_ITEM(&conIdsToRemove, conIdsToRemove.count - 1);

              break;
            }
          }
        }

        for (uint32_t i = 0; i < incomingNeuronIds.count; i++) {
          uint32_t outgoing = NEAT_findNeuron(g, outgoingNeuronId);
          uint32_t incoming = NEAT_findNeuron(g, incomingNeuronIds.items[i]);

          enum NEAT_ConnectionKind kind = g->neurons.items[outgoing].layer >
                                              g->neurons.items[incoming].layer
                                            ? NEAT_CON_KIND_FORWARD
                                            : NEAT_CON_KIND_RECURRENT;

          NEAT_createConnection(g, kind, outgoingNeuronId,
                                incomingNeuronIds.items[i], false, ctx);
        }

        DA_FREE(&conIdsToRemove);
        DA_FREE(&incomingNeuronIds);
      }

      DA_DELETE_ITEM(&g->neurons, neuronToRemove);

      goto Exit;
    }
  }

Exit:
  DA_FREE(&removableNeurons);
  return;
}

int main(void) {
  srand(time(NULL));

  struct NEAT_Parameters p = {
    .inputs = 1,
    .outputs = 1,
    .populationSize = 7,
    .allowRecurrent = false,
    .parentMutationProbability = 0.01f,
    .childMutationProbability = 0.3f,
    .maxMutationsPerGeneration = 3,
    .conToggleProbability = 0.0f,
    .weightNudgeProbability = 0.0f,
    .weightRandomizeProbability = 0.0f,
    .connectionAddProbability = 0.0f,
    .neuronAddProbability = 1.0f,
    .connectionDeleteProbability = 0.0f,
    .neuronDeleteProbability = 0.0f,
    .elitismProportion = 0.2f,
    .sexualProportion = 0.5f,
    .interspeciesProbability = 0.1f,
    .initialSpeciesTarget = 10,
    .initialSpeciationThreshold = 1.5f,
    .improvementDeadline = 15,
  };

  struct NEAT_Context ctx = NEAT_constructPopulation(&p);

  //DA_FREE(&ctx.population[0].connections);

  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 2, 0, false,
  //                      &ctx);

  //ctx.population[0].connections.items[0].enabled = false;

  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD,
  //                      ctx.population[0].nextNeuronId - 1, 1, false, &ctx);

  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_RECURRENT,
  //                      ctx.population[0].nextNeuronId - 1, 2, false, &ctx);

  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 2,
  //                      ctx.population[0].nextNeuronId - 1, false, &ctx);
  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3,
  //                      ctx.population[0].nextNeuronId - 1, false, &ctx);
  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 4,
  //                      ctx.population[0].nextNeuronId - 1, false, &ctx);
  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 5,
  //                      ctx.population[0].nextNeuronId - 1, false, &ctx);

  //NEAT_mutate(&ctx.population[0], NEAT_COMPLEXIFY, &ctx);
  NEAT_layer(&ctx);

  NEAT_printNetwork(&ctx.population[0]);

  //for (uint32_t i = 0; i < ctx.populationSize; i++) {
  //  NEAT_printNetwork(&ctx.population[i]);
  //  printf("Next neuron ID: %d\n", ctx.population[i].nextNeuronId);
  //}

  uint32_t factor = 120;
  uint32_t width = 16 * factor;
  uint32_t height = 9 * factor;
  SetConfigFlags(FLAG_MSAA_4X_HINT);
  InitWindow(width, height, "NEAT");
  SetWindowState(FLAG_WINDOW_RESIZABLE | FLAG_WINDOW_MAXIMIZED);

  uint32_t bgColour = 0xff181818;

  uint32_t i = 0;
  float t = 0;
  bool paused = true;
  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE)) {
      paused = !paused;
    }

    if (IsKeyPressed(KEY_R)) {
      NEAT_cleanup(&ctx);

      uint32_t inps = ((float)rand() / (float)RAND_MAX) * (5 - 1) + 1;
      uint32_t outs = ((float)rand() / (float)RAND_MAX) * (7 - 1) + 1;

      float conNudge = ((float)rand() / (float)RAND_MAX);
      float conRand = ((float)rand() / (float)RAND_MAX) * (1.0f - conNudge);
      float conNew =
        ((float)rand() / (float)RAND_MAX) * (1.0f - (conNudge + conRand));
      float conToggle = ((float)rand() / (float)RAND_MAX) *
                        (1.0f - (conNudge + conRand + conNew));

      conToggle = 0.0f;

      float nNew = (1.0f - (conNudge + conRand + conNew + conToggle));

      printf("%f\n", conNudge + conRand + conNew + nNew + conToggle);

      ctx = NEAT_constructPopulation(&(struct NEAT_Parameters){
        .inputs = inps,
        .outputs = outs,
        .populationSize = 7,
        .allowRecurrent = false,
        .parentMutationProbability = 0.01f,
        .childMutationProbability = 0.3f,
        .maxMutationsPerGeneration = 3,

        .conToggleProbability = conToggle,
        .weightNudgeProbability = conNudge,
        .weightRandomizeProbability = conRand,
        .connectionAddProbability = conNew,
        .neuronAddProbability = nNew,
        .connectionDeleteProbability = conNew,
        .neuronDeleteProbability = nNew,

        .elitismProportion = 0.2f,
        .sexualProportion = 0.5f,
        .interspeciesProbability = 0.1f,
        .initialSpeciesTarget = 10,
        .initialSpeciationThreshold = 1.5f,
        .improvementDeadline = 15,
      });

      NEAT_layer(&ctx);
      i = 0;
      t = 0;
    }

    if (!paused)
      t += GetFrameTime();

    if (t >= 0.0f && i < 1000 && !paused) {
      //uint32_t con = rand() % ctx.population[0].connections.count;
      //NEAT_splitConnection(&ctx.population[0], con, &ctx);
      //NEAT_layer(&ctx);
      uint8_t temp = rand() % 2;
      enum NEAT_Phase p = temp ? NEAT_COMPLEXIFY : NEAT_PRUNE;
      NEAT_mutate(&ctx.population[0], p, &ctx);
      NEAT_layer(&ctx);
      i++;
      t = 0.0f;
    }

    width = GetRenderWidth();
    height = GetRenderHeight();
    ClearBackground(*(Color *)&bgColour);
    BeginDrawing();
    drawNetwork(&ctx.population[0], 0.05f * width, 0.05f * height, 0.9f * width,
                0.9f * height);

    EndDrawing();
  }

  CloseWindow();

  NEAT_cleanup(&ctx);

  return 0;
}
