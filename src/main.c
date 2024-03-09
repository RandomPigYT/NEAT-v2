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

        if (g->connections.items[connections.items[k]].kind ==
            NEAT_CON_KIND_RECURRENT) {
          colour = weight >= 0 ? ORANGE : BLUE;

          colour = ColorAlphaBlend(
            colour, CLITERAL(Color){ .a = alpha, .r = 35, .g = 35, .b = 35 },
            WHITE);

          if (g->neurons.items[toNeurons.items[k]].layer == i) {
            if (toNeurons.items[k] == neuronsInLayer.items[j]) {
              Vector2 ringCentre = {
                .x = cx1,
                .y = cy1 - 2 * radius,
              };

              float outRad = radius * 2;
              float inRad = outRad - thickness;

              DrawRing(ringCentre, inRad, outRad, 0.0f, 360.0f, 100, colour);

            } else {
              DrawSplineBezierQuadratic(
                (Vector2[]){
                  CLITERAL(Vector2){
                    cx1 + (g->neurons.items[toNeurons.items[k]].kind ==
                               NEAT_NEURON_KIND_OUTPUT
                             ? -1
                             : 1) *
                            radius,
                    cy1 },
                  CLITERAL(Vector2){
                    cx1 + (g->neurons.items[toNeurons.items[k]].kind ==
                               NEAT_NEURON_KIND_OUTPUT
                             ? -1
                             : 1) *
                            (w * 0.05),
                    (cy1 + cy2) / 2.0f },
                  CLITERAL(Vector2){
                    cx2 + (g->neurons.items[toNeurons.items[k]].kind ==
                               NEAT_NEURON_KIND_OUTPUT
                             ? -1
                             : 1) *
                            radius,
                    cy2 },
                },
                3, thickness, colour);
            }
          } else {
            DrawLineBezier(CLITERAL(Vector2){ cx1, cy1 },
                           CLITERAL(Vector2){ cx2, cy2 }, thickness, colour);
          }

        } else {
          DrawLineEx(CLITERAL(Vector2){ cx1, cy1 },
                     CLITERAL(Vector2){ cx2, cy2 }, thickness, colour);
        }

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

struct NEAT_Genome NEAT_crossover(const struct NEAT_Genome *p1,
                                  const struct NEAT_Genome *p2,
                                  struct NEAT_Context *ctx) {
  // p1 always has fitness greater than or equal to p2
  if (p1->fitness < p2->fitness) {
    const struct NEAT_Genome *temp = p1;
    p1 = p2;
    p2 = temp;
  }

  struct NEAT_Genome child = NEAT_constructNetwork(false, ctx);

  DA_CREATE(uint32_t) p1DisjointGenes = { 0 };
  DA_CREATE(uint32_t) p2DisjointGenes = { 0 };

  // Matching genes
  for (uint32_t i = 0; i < p1->connections.count; i++) {
    bool match = false;
    uint32_t matchIndex = 0;
    for (uint32_t j = 0; j < p2->connections.count; j++) {
      if (p2->connections.items[j].innovation ==
          p1->connections.items[i].innovation) {
        match = true;
        matchIndex = j;

        break;
      }
    }

    if (!match) {
      DA_APPEND(&p1DisjointGenes, i);
      continue;
    }

    uint32_t currentIndex = child.connections.count;

    NEAT_createConnection(&child, p1->connections.items[i].kind,
                          p1->connections.items[i].to,
                          p1->connections.items[i].from, true, true, NULL);

    child.connections.items[currentIndex].innovation =
      p1->connections.items[i].innovation;

    float random = (float)rand() / (float)RAND_MAX;
    if (random <= 0.5f) {
      child.connections.items[currentIndex].weight =
        p1->connections.items[i].weight;
      child.connections.items[currentIndex].enabled =
        p1->connections.items[i].enabled;
    } else {
      child.connections.items[currentIndex].weight =
        p2->connections.items[matchIndex].weight;
      child.connections.items[currentIndex].enabled =
        p2->connections.items[matchIndex].enabled;
    }
  }

  // Non-matching genes
  bool fitnessEqual = p1->fitness == p2->fitness ? true : false;
  if (fitnessEqual) {
    for (uint32_t i = 0; i < p2->connections.count; i++) {
      bool match = false;
      for (uint32_t j = 0; j < p1->connections.count; j++) {
        if (p1->connections.items[j].innovation ==
            p2->connections.items[i].innovation) {
          match = true;

          break;
        }
      }

      if (!match) {
        DA_APPEND(&p2DisjointGenes, i);
      }
    }
  }

  for (uint32_t i = 0; i < p1DisjointGenes.count; i++) {
    bool shouldInheritGene = true;
    if (fitnessEqual) {
      float random = (float)rand() / (float)RAND_MAX;
      if (random > 0.5f) {
        shouldInheritGene = false;
      }
    }

    if (shouldInheritGene) {
      uint32_t p1ConIndex = p1DisjointGenes.items[i];
      uint32_t childConIndex = child.connections.count;
      NEAT_createConnection(&child, p1->connections.items[p1ConIndex].kind,
                            p1->connections.items[p1ConIndex].to,
                            p1->connections.items[p1ConIndex].from, true,
                            p1->connections.items[p1ConIndex].enabled, NULL);

      child.connections.items[childConIndex].innovation =
        p1->connections.items[p1ConIndex].innovation;
      child.connections.items[childConIndex].weight =
        p1->connections.items[p1ConIndex].weight;
    }
  }

  if (fitnessEqual) {
    for (uint32_t i = 0; i < p2DisjointGenes.count; i++) {
      bool shouldInheritGene = true;

      float random = (float)rand() / (float)RAND_MAX;
      if (random > 0.5f) {
        shouldInheritGene = false;
      }

      if (shouldInheritGene) {
        uint32_t p2ConIndex = p2DisjointGenes.items[i];
        uint32_t childConIndex = child.connections.count;
        NEAT_createConnection(&child, p2->connections.items[p2ConIndex].kind,
                              p2->connections.items[p2ConIndex].to,
                              p2->connections.items[p2ConIndex].from, true,
                              p2->connections.items[p2ConIndex].enabled, NULL);

        child.connections.items[childConIndex].innovation =
          p2->connections.items[p2ConIndex].innovation;
        child.connections.items[childConIndex].weight =
          p2->connections.items[p2ConIndex].weight;
      }
    }
  }

  DA_FREE(&p1DisjointGenes);
  DA_FREE(&p2DisjointGenes);

  return child;
}

int main(void) {
  //srand(time(NULL));
  srand(69420);

  struct NEAT_Parameters p = {
    .inputs = 2,
    .outputs = 1,
    .populationSize = 7,

    .allowRecurrent = true,
    .recurrentProbability = 0.1f,

    .parentMutationProbability = 0.01f,
    .childMutationProbability = 0.3f,
    .maxMutationsPerGeneration = 3,

    .conToggleProbability = 0.25f,
    .weightNudgeProbability = 0.25f,
    .weightRandomizeProbability = 0.25f,

    .connectionAddProbability = 0.125f,
    .neuronAddProbability = 0.125f,
    .connectionDeleteProbability = 0.125f,
    .neuronDeleteProbability = 0.125f,

    .elitismProportion = 0.2f,
    .sexualProportion = 0.5f,
    .interspeciesProbability = 0.1f,

    .initialSpeciesTarget = 10,
    .initialSpeciationThreshold = 1.5f,
    .improvementDeadline = 15,
  };

  //struct NEAT_Parameters p = {
  //  .inputs = 2,
  //  .outputs = 4,
  //  .populationSize = 7,

  //  .allowRecurrent = true,
  //  .recurrentProbability = 0.1f,

  //  .parentMutationProbability = 0.01f,
  //  .childMutationProbability = 0.3f,
  //  .maxMutationsPerGeneration = 3,

  //  .conToggleProbability = 0.0f,
  //  .weightNudgeProbability = 0.0f,
  //  .weightRandomizeProbability = 0.0f,

  //  .connectionAddProbability = 0.0f,
  //  .neuronAddProbability = 1.0f,
  //  .connectionDeleteProbability = 0.0f,
  //  .neuronDeleteProbability = 0.0f,

  //  .elitismProportion = 0.2f,
  //  .sexualProportion = 0.5f,
  //  .interspeciesProbability = 0.1f,

  //  .initialSpeciesTarget = 10,
  //  .initialSpeciationThreshold = 1.5f,
  //  .improvementDeadline = 15,
  //};

  struct NEAT_Context ctx = NEAT_constructPopulation(&p);

  //NEAT_mutate(&ctx.population[0], NEAT_COMPLEXIFY, &ctx);

  DA_FREE(&ctx.population[0].connections);
  DA_FREE(&ctx.population[1].connections);

  DA_FREE(&ctx.history);
  ctx.globalInnovation = 0;

  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3, 0, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3, 1, true,
                        false, &ctx);
  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3, 2, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 4, 1, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 3, 4, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_FORWARD, 4, 0, true,
                        true, &ctx);

  ctx.population[0]
    .connections.items[ctx.population[0].connections.count - 1]
    .innovation = 7;

  ctx.globalInnovation = 8;

  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 3, 0, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 3, 1, true,
                        false, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 3, 2, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 4, 1, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 3, 4, true,
                        false, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 5, 4, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 3, 5, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 4, 2, true,
                        true, &ctx);
  NEAT_createConnection(&ctx.population[1], NEAT_CON_KIND_FORWARD, 5, 0, true,
                        true, &ctx);

  ctx.population[0].fitness = 1.0f;
  ctx.population[1].fitness = ctx.population[0].fitness - 1;
  ctx.population[0] =
    NEAT_crossover(&ctx.population[0], &ctx.population[1], &ctx);

  NEAT_layer(&ctx);
  NEAT_printNetwork(&ctx.population[0]);

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

      //printf("%f\n", conNudge + conRand + conNew + nNew + conToggle);
      float recProb = (float)rand() / (float)RAND_MAX;

      ctx = NEAT_constructPopulation(&(struct NEAT_Parameters){
        .inputs = inps,
        .outputs = outs,
        .populationSize = 7,

        .allowRecurrent = true,
        .recurrentProbability = recProb,

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

    if (t >= 0.0f /* && i < 1000 */ && !paused) {
      //uint32_t con = rand() % ctx.population[0].connections.count;
      //NEAT_splitConnection(&ctx.population[0], con, &ctx);
      //NEAT_layer(&ctx);

      DA_FREE(&ctx.history);
      //printf("%d\n", i);

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

  //NEAT_printNetwork(&ctx.population[0]);

  NEAT_cleanup(&ctx);

  return 0;
}
