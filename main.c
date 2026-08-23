#include <immintrin.h>
#include <raylib.h>
#include <stdio.h>
#include <math.h>

float fast_sum( __m256 ax0) {
    __m128 lo = _mm256_castps256_ps128(ax0); // [a, b, c, d]
    __m128 hi = _mm256_extractf128_ps(ax0, 1); // [e, f, g ,h]

    __m128 s = _mm_add_ps(lo, hi); // [a + e, b + f, c + g, d + h]
    s = _mm_hadd_ps(s, s); // [a+e+b+f, c+g+d+h, a+e+b+f, c+g+d+h]
    s = _mm_hadd_ps(s, s); // [a+..h, a..h, a..h, a..h]

    float ax_sum = _mm_cvtss_f32(s);
    return ax_sum;
}

#define TRAIL_LEN 400

typedef struct {
    float x[TRAIL_LEN];
    float y[TRAIL_LEN];
    int head;
    int count;
} Trail;

void trail_push(Trail *t, float x, float y) {
    t->head = (t->head + 1) % TRAIL_LEN;
    t->x[t->head] = x;
    t->y[t->head] = y;
    if (t->count < TRAIL_LEN) t->count++;
}

void render_trails(Trail trails[], float masses[8], int n, float center_x, float center_y, float scale) {
    for (int i = 0; i < n; i++) {
        Trail *t = &trails[i];
        for (int j = 0; j < t->count; j++) {
            if(masses[i] == 0) break;
            int idx = (t->head - j + TRAIL_LEN) % TRAIL_LEN;
            float px = center_x + t->x[idx] * scale;
            float py = center_y - t->y[idx] * scale;
            float alpha = 1.0f - (float)j / (float)t->count;
            DrawCircle((int)px, (int)py, 3.0f, Fade(SKYBLUE, alpha * 0.6f));
        }
    }
}

void render_points(const float xs[], const float ys[], float masses[8], int count, float center_x, float center_y, float scale) {
    for (int i = 0; i < count; i++) {
        if(masses[i] == 0) continue;
        float px = center_x + xs[i] * scale;
        float py = center_y - ys[i] * scale;

        DrawCircle((int)px, (int)py, 8.0f, BLUE);
        DrawCircleLines((int)px, (int)py, 8.0f, SKYBLUE);
        DrawText(
            TextFormat("%d", i),
            (int)px - MeasureText(TextFormat("%d", i), 12) / 2,
            (int)py - 30,
            12,
            SKYBLUE
        );
    }
}

void get_instantenout_acceletation(float s_x, float s_y, __m256 x, __m256 y, __m256 m, __m256 eta, float result[2]) {
    __m256 x0 = _mm256_set1_ps(s_x);
    __m256 y0 = _mm256_set1_ps(s_y);

    __m256 rx = _mm256_sub_ps(x, x0);
    __m256 ry = _mm256_sub_ps(y, y0);

    __m256 rx2 = _mm256_mul_ps(rx, rx);
    __m256 ry2 = _mm256_mul_ps(ry, ry);
    
    __m256 r2 = _mm256_add_ps(rx2, ry2);
    __m256 r_eta = _mm256_add_ps(r2, eta);
    
    __m256 r_half = _mm256_sqrt_ps(r_eta);
    __m256 r_32 = _mm256_mul_ps(r_eta, r_half);

    __m256 rx_r = _mm256_div_ps(rx, r_32);
    __m256 ax0 = _mm256_mul_ps(m, rx_r);

    __m256 ry_r = _mm256_div_ps(ry, r_32);
    __m256 ay0 = _mm256_mul_ps(m, ry_r);

    float ax_inst = fast_sum(ax0);
    float ay_inst = fast_sum(ay0);

    result[0] = ax_inst;
    result[1] = ay_inst;
}


void get_instantenout_position(float X[8], float Y[8], float Vx[8], float Vy[8], __m256 m, __m256 delta_t, __m256 eta, float ax_inst[8], float ay_inst[8]) {
    __m256 x = _mm256_loadu_ps(X);
    __m256 y = _mm256_loadu_ps(Y);
    __m256 vx = _mm256_loadu_ps(Vx);
    __m256 vy = _mm256_loadu_ps(Vy);
    
    float buffer[2];
    for(int i=0; i<8; i++) {
        get_instantenout_acceletation(X[i], Y[i], x, y, m, eta, buffer);
        ax_inst[i] = buffer[0];
        ay_inst[i] = buffer[1];
    }

    __m256 ax_inst_avx = _mm256_loadu_ps(ax_inst);
    __m256 ay_inst_avx = _mm256_loadu_ps(ay_inst);

    __m256 delta_vx = _mm256_mul_ps(ax_inst_avx, delta_t);
    __m256 delta_vy = _mm256_mul_ps(ay_inst_avx, delta_t);

    vx = _mm256_add_ps(vx, delta_vx);
    vy = _mm256_add_ps(vy, delta_vy);
    
    x = _mm256_add_ps(
        x,
        _mm256_mul_ps(vx, delta_t)
    );

    y = _mm256_add_ps(
        y,
        _mm256_mul_ps(vy, delta_t)
    );
    
    _mm256_storeu_ps(X, x);
    _mm256_storeu_ps(Y, y);
    _mm256_storeu_ps(Vx, vx);
    _mm256_storeu_ps(Vy, vy);
}

int main() {
    // __m256 m0 = _mm256_set1_ps(1.0f);

    const float ETA = 2.0f;
    const float SUN_MASS = 400000.0f;

    float v_x = 0, v_y = 0, dt = 1.0f/60.0f;
    float s_x = 0.0f, s_y = 0.0f;
    __m256 eta = _mm256_set1_ps(ETA);
    __m256 delta_t = _mm256_set1_ps(dt);

    InitWindow(100, 100, "SIMD Physics");

    int monitor = GetCurrentMonitor();
    int screen_width = GetMonitorWidth(monitor);
    int screen_height = GetMonitorHeight(monitor);

    SetWindowState(FLAG_WINDOW_UNDECORATED);
    SetWindowSize(screen_width, screen_height);
    SetWindowPosition(0, 0);
    SetTargetFPS(60);

    // index 0 is the sun, sitting at the origin; the rest are planets.
    // Planet 4 sits further out (r=30, not 10) so the sun's tidal pull at
    // its orbit is weak enough for a modest planet mass to still hold onto
    // a moon — see the moon setup below for why this matters.
    float X[8] =      {0.0f,     4.0f,     10.0f,   -6.0f,   -30.0f,    -31.0f,      7.0f,       -8.0f};
    float Y[8] =      {0.0f,     0.0f,     10.0f,   6.0f,    0.0f,      0.0f,      -5.0f,      -2.0f};
    float Vx[8] =     {0.0f,     0.0f,     0.0f,    0.0f,    0.0f,      0.0f,       0.0f,       0.0f};
    float Vy[8] =     {0.0f,     0.0f,     0.0f,    0.0f,    0.0f,      0.0f,       0.0f,       0.0f};
    float masses[8] = {SUN_MASS, 0.0f,     0.0f,    0.0f,    300.0f,    3.0f,       0.0f,       0.0f};
    // Deviation from the circular-orbit speed for each planet: 1.0 would be
    // a perfect circle. <1 pulls perihelion closer to the sun, >1 pushes
    // aphelion further out — both give a visibly eccentric ellipse as long
    // as we stay under the ~1.41 escape-velocity factor.
    float eccentricity_factor[8] = {0.0f, 0.75f, 1.2f, 0.85f, 1.15f, 0.65f, 1.3f, 0.9f};
    float buffer1[8], buffer2[8];

    Trail trails[8] = {0};

    // Give each planet a tangential velocity sized for a circular orbit
    // around the sun (scaled by eccentricity_factor to make it elliptical),
    // so gravity curves the path instead of pulling it straight in:
    // v = factor * r * sqrt(SUN_MASS) / (r^2 + eta)^0.75.
    for (int i = 1; i < 8; i++) {
        float r2 = X[i] * X[i] + Y[i] * Y[i];
        float r = sqrtf(r2);
        float v_mag = eccentricity_factor[i] * r * sqrtf(SUN_MASS) / powf(r2 + ETA, 0.75f);

        Vx[i] = -Y[i] / r * v_mag;
        Vy[i] = X[i] / r * v_mag;
    }

    __m256 m = _mm256_loadu_ps(masses);

    while (!WindowShouldClose())  {
        // get_instantenout_acceletation(s_x, s_y, x, y, m, eta, buffer);
        get_instantenout_position(X, Y, Vx, Vy, m, delta_t, eta, buffer1, buffer2);

        for (int i = 0; i < 8; i++) {
            // trail_push(&trails[i], X[i], Y[i]);
        }

        BeginDrawing();

        ClearBackground(BLACK);

        /*
         * Convert physics coordinates into
         * screen coordinates.
         */
        float scale = 14.0f; // shrunk from 60 so the planet's r~30-60 orbit fits on screen

        float center_x = screen_width / 2.0f;
        float center_y = screen_height / 2.0f;



        /*
         * Draw coordinate axes
         */
        DrawLine(
            0,
            (int)center_y,
            screen_width,
            (int)center_y,
            DARKGRAY
        );

        DrawLine(
            (int)center_x,
            0,
            (int)center_x,
            screen_height,
            DARKGRAY
        );

        render_trails(trails, masses, 8, center_x, center_y, scale);
        render_points(X, Y, masses, 8, center_x, center_y, scale);

        /*

         * Header
         */
        DrawText("SIMD Physics — Gravity Simulation", 20, 15, 22, RAYWHITE);
        DrawFPS(screen_width - 90, 15);

        /*
         * Debug panel
         */
        int panel_x = 15, panel_y = 55, panel_w = 190, panel_h = 110;
        DrawRectangle(panel_x, panel_y, panel_w, panel_h, Fade(BLACK, 0.55f));
        DrawRectangleLines(panel_x, panel_y, panel_w, panel_h, Fade(GRAY, 0.8f));

        DrawText(TextFormat("x:  %8.3f", s_x), panel_x + 12, panel_y + 10, 18, WHITE);
        DrawText(TextFormat("y:  %8.3f", s_y), panel_x + 12, panel_y + 35, 18, WHITE);
        DrawText(TextFormat("vx: %8.3f", v_x), panel_x + 12, panel_y + 60, 18, GREEN);
        DrawText(TextFormat("vy: %8.3f", v_y), panel_x + 12, panel_y + 85, 18, GREEN);

        EndDrawing();
    }
    CloseWindow();
    return 0;
}
