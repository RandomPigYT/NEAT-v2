#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <raylib.h>

#define NEAT_H_IMPLEMENTATION
#include "../include/NEAT.h"

#define NEAT_VISUALISE_H_IMPLEMENTATION
#include "../include/NEAT-Visualise.h"

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
  //NEAT_createConnection(&ctx.population[0], NEAT_CON_KIND_RECURRENT, 0, 4, true,
  //                      true, &ctx);

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
  ctx.population[1].fitness = ctx.population[0].fitness;
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
    NEAT_Visualise_drawNetwork(&ctx.population[0], 0.05f * width,
                               0.05f * height, 0.9f * width, 0.9f * height);

    EndDrawing();
  }

  CloseWindow();

  //NEAT_printNetwork(&ctx.population[0]);

  NEAT_cleanup(&ctx);

  return 0;
}
