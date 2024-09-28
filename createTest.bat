g++ -c -g -O3 test.cpp -IC:/Users/aaron/CODING/cpp_libs/SFML-2.6.1/include -IC:/Users/aaron/CODING/cpp_libs/glm-1.0.1-light -DSFML_STATIC
g++ test.o -o test -Wall -Wextra -LC:\Users\aaron\CODING\cpp_libs\SFML-2.6.1\lib -lsfml-graphics-s -lsfml-window-s -lsfml-system-s -lopengl32 -lwinmm -lgdi32 -lfreetype -static
del test.o