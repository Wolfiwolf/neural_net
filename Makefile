TARGET = neural_net_example


SRC_DIR = src
BLD_DIR = build

SRCS = $(shell find $(SRC_DIR) -name "*.cpp")
INCS = $(shell find $(SRC_DIR) -name "*.hpp")
OBJS = $(SRCS:$(SRC_DIR)/%=$(BLD_DIR)/%.o)

$(TARGET): $(OBJS) 
	g++ -o $@ $^

$(BLD_DIR)/%.o: $(SRC_DIR)/% $(INCS)
	@mkdir -p $(@D)
	g++ -c $< -o $@

clean:
	rm -r $(BLD_DIR) $(TARGET)
