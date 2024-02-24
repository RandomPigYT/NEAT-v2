CC:=gcc
LD:=gcc

CFLAGS:=-Wall -Wextra -g -std=gnu17 -I.
LDFLAGS:=

BIN:=bin
OBJ:=obj
SRC:=src
INCLUDE:=include

VALDIR:=valdir


TARGET:=$(BIN)/NEAT_gym
VALGRIND_OUT:=$(VALDIR)/val_out.txt

SRCS:=$(shell find $(SRC) -type  f -name "*.c")
OBJS:=$(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))
INCLUDES:=$(shell find $(INCLUDE) -type f -name "*.h") $(shell find $(SRC) -type f -name "*.h")

DIRS:=$(patsubst $(SRC)/%, $(OBJ)/%, $(shell find $(SRC)/ -mindepth 1 -type d))

CREATE_DIR_COMMAND:=./dirs.sh

.PHONY: all clean dirs valgrind format run

all: dirs $(TARGET)

$(TARGET): $(OBJS)
	@echo
	@echo building $(TARGET)
	@$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)
	@echo built $(TARGET)

$(OBJ)/%.o: $(SRC)/%.c $(INCLUDES)
	@echo building $@
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo built $@
	@echo

dirs:
	@mkdir -p $(BIN)
	@mkdir -p $(OBJ)
	@$(CREATE_DIR_COMMAND) $(DIRS)

clean:
	-@rm -rf $(OBJ)
	-@rm -rf $(BIN)

run: dirs $(TARGET)
	@./$(TARGET)

valgrind:
	@valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=$(VALGRIND_OUT) \
         ./$(TARGET)	"test"

format:
	@clang-format $(SRCS) $(INCLUDES) --style=Google -i

