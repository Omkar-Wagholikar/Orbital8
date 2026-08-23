CC = gcc
CFLAGS = -Wall -Wextra -O2 -march=native
LDFLAGS = -lraylib -lm -lX11

TARGET = main

all: $(TARGET)

$(TARGET): main.c
	$(CC) $(CFLAGS) main.c -o $(TARGET) $(LDFLAGS)

run: FORCE
	$(MAKE) $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean FORCE

FORCE:
