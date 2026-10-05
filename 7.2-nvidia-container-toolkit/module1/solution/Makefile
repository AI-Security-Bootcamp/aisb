# Makefile to compile poc.cpp to poc.so

# Compiler and flags
CC = gcc
CXX = g++
CXXFLAGS = -fPIC -shared -O2 -Wall
LDFLAGS = 

# Source and target
SOURCE = poc.c
TARGET = poc.so

# Default target
all: $(TARGET) docker

# Compile poc.cpp to poc.so
$(TARGET): $(SOURCE)
	$(CC) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

# Clean target
clean:
	rm -f $(TARGET)

# Force rebuild
rebuild: clean all

docker:
	docker build -t nvidia-ctk-image .

# Mark targets as phony
.PHONY: all clean rebuild docker
