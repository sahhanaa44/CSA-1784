/*
 * ============================================================
 *  Intelligent Elder Fall Detection System
 *  TinyML + LSTM on ESP32
 *
 *  Hardware:
 *    - ESP32 DevKit (any variant)
 *    - MPU6050 (IMU) → I2C: SDA=GPIO21, SCL=GPIO22
 *    - Buzzer        → GPIO 25
 *    - LED indicator → GPIO 26
 *
 *  Modules:
 *    1. Motion Monitoring  – MPU6050 reads accel + gyro
 *    2. LSTM Inference     – TensorFlow Lite Micro LSTM model
 *    3. Emergency Alert    – Telegram Bot notification over WiFi
 *
 *  Upload via: Arduino IDE or PlatformIO (see README)
 * ============================================================
 */

#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "lstm_model.h"   // <-- generated model header (see README)

// ─── User Configuration ───────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* BOT_TOKEN     = "YOUR_TELEGRAM_BOT_TOKEN";   // from @BotFather
const char* CHAT_ID       = "YOUR_TELEGRAM_CHAT_ID";     // recipient chat id

// ─── Pin Definitions ─────────────────────────────────────────────────────────
#define MPU_SDA       21
#define MPU_SCL       22
#define BUZZER_PIN    25
#define LED_PIN       26

// ─── MPU6050 Registers ───────────────────────────────────────────────────────
#define MPU6050_ADDR  0x68
#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B
#define GYRO_XOUT_H   0x43

// ─── LSTM Inference Parameters ───────────────────────────────────────────────
#define SEQUENCE_LEN   50    // timesteps fed to LSTM (50 × 20 ms = 1-second window)
#define NUM_FEATURES   6     // ax, ay, az, gx, gy, gz
#define TENSOR_ARENA_SIZE (80 * 1024)  // 80 KB; tune down if you run out of RAM

// ─── Detection Thresholds ────────────────────────────────────────────────────
#define FALL_THRESHOLD       0.80f   // LSTM output confidence to declare fall
#define IMPACT_ACCEL_G       2.5f    // pre-filter: raw impact magnitude (g)
#define COOLDOWN_MS          8000    // suppress repeat alerts for 8 s

// ─── Globals ─────────────────────────────────────────────────────────────────
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::AllOpsResolver      resolver;
const tflite::Model*        model      = nullptr;
tflite::MicroInterpreter*   interpreter = nullptr;
TfLiteTensor*               input_tensor = nullptr;
TfLiteTensor*               output_tensor = nullptr;

float imu_buffer[SEQUENCE_LEN][NUM_FEATURES];  // sliding window
int   buffer_idx   = 0;
bool  buffer_ready = false;

unsigned long last_alert_ms = 0;

// ─── MPU6050 Helpers ─────────────────────────────────────────────────────────
void mpu_init() {
  Wire.begin(MPU_SDA, MPU_SCL);
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(PWR_MGMT_1);
  Wire.write(0x00);   // wake up
  Wire.endTransmission(true);
  delay(100);
  Serial.println("[MPU6050] Initialized");
}

struct ImuReading { float ax, ay, az, gx, gy, gz; };

ImuReading mpu_read() {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 14, true);

  int16_t raw_ax = (Wire.read() << 8) | Wire.read();
  int16_t raw_ay = (Wire.read() << 8) | Wire.read();
  int16_t raw_az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();  // skip temperature
  int16_t raw_gx = (Wire.read() << 8) | Wire.read();
  int16_t raw_gy = (Wire.read() << 8) | Wire.read();
  int16_t raw_gz = (Wire.read() << 8) | Wire.read();

  ImuReading r;
  r.ax = raw_ax / 16384.0f;   // ±2 g  scale
  r.ay = raw_ay / 16384.0f;
  r.az = raw_az / 16384.0f;
  r.gx = raw_gx / 131.0f;    // ±250 °/s scale
  r.gy = raw_gy / 131.0f;
  r.gz = raw_gz / 131.0f;
  return r;
}

// ─── TFLite Model Init ───────────────────────────────────────────────────────
void model_init() {
  model = tflite::GetModel(lstm_model_tflite);  // from lstm_model.h
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TFLite] Model schema version mismatch!");
    while (true);
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  TfLiteStatus status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    Serial.println("[TFLite] AllocateTensors() failed!");
    while (true);
  }

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.printf("[TFLite] Model loaded. Input shape: [1,%d,%d]\n",
                SEQUENCE_LEN, NUM_FEATURES);
}

// ─── Inference ───────────────────────────────────────────────────────────────
/*
 * Copies the sliding window into the model input tensor,
 * runs inference, returns fall probability [0..1].
 */
float run_inference() {
  // Fill input tensor  shape = [1, SEQUENCE_LEN, NUM_FEATURES]
  for (int t = 0; t < SEQUENCE_LEN; t++) {
    int src = (buffer_idx + t) % SEQUENCE_LEN;   // correct circular index
    for (int f = 0; f < NUM_FEATURES; f++) {
      int flat = t * NUM_FEATURES + f;
      if (input_tensor->type == kTfLiteFloat32) {
        input_tensor->data.f[flat] = imu_buffer[src][f];
      } else if (input_tensor->type == kTfLiteInt8) {
        // Quantized model path
        float scale     = input_tensor->params.scale;
        int32_t zero_pt = input_tensor->params.zero_point;
        input_tensor->data.int8[flat] =
            (int8_t)(imu_buffer[src][f] / scale + zero_pt);
      }
    }
  }

  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("[TFLite] Invoke() failed!");
    return 0.0f;
  }

  float prob;
  if (output_tensor->type == kTfLiteFloat32) {
    prob = output_tensor->data.f[1];   // index 1 = "fall" class
  } else {
    float scale     = output_tensor->params.scale;
    int32_t zero_pt = output_tensor->params.zero_point;
    prob = (output_tensor->data.int8[1] - zero_pt) * scale;
  }
  return prob;
}

// ─── WiFi ────────────────────────────────────────────────────────────────────
void wifi_connect() {
  Serial.print("[WiFi] Connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000) {
    delay(500); Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Connected – IP: %s\n",
                  WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] Connection failed – alerts will be skipped");
  }
}

// ─── Telegram Alert ──────────────────────────────────────────────────────────
void send_telegram_alert(float confidence) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[Alert] WiFi not connected, skipping Telegram");
    return;
  }

  String url = String("https://api.telegram.org/bot") + BOT_TOKEN + "/sendMessage";
  String msg = String("⚠️ *FALL DETECTED!*\n\n") +
               "An elder may have fallen.\n" +
               "Confidence: " + String(confidence * 100, 1) + "%\n" +
               "Please check immediately!";

  StaticJsonDocument<256> doc;
  doc["chat_id"]    = CHAT_ID;
  doc["text"]       = msg;
  doc["parse_mode"] = "Markdown";

  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  Serial.printf("[Telegram] Response code: %d\n", code);
  http.end();
}

// ─── Alert Sequence ──────────────────────────────────────────────────────────
void trigger_alert(float confidence) {
  unsigned long now = millis();
  if (now - last_alert_ms < COOLDOWN_MS) return;
  last_alert_ms = now;

  Serial.printf("[ALERT] Fall detected! Confidence = %.2f\n", confidence);

  // Buzzer: 3 short beeps
  for (int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(150);
  }

  // LED: rapid flash
  for (int i = 0; i < 10; i++) {
    digitalWrite(LED_PIN, HIGH); delay(100);
    digitalWrite(LED_PIN, LOW);  delay(100);
  }

  send_telegram_alert(confidence);
}

// ─── setup() ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== Elder Fall Detection System Booting ===");

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, HIGH);   // on during init

  mpu_init();
  model_init();
  wifi_connect();

  memset(imu_buffer, 0, sizeof(imu_buffer));
  buffer_idx   = 0;
  buffer_ready = false;

  digitalWrite(LED_PIN, LOW);
  Serial.println("[System] Ready – monitoring started\n");
}

// ─── loop() ──────────────────────────────────────────────────────────────────
void loop() {
  static unsigned long last_sample_ms = 0;
  unsigned long now = millis();

  // Sample at ~50 Hz (every 20 ms)
  if (now - last_sample_ms < 20) return;
  last_sample_ms = now;

  ImuReading r = mpu_read();

  // Store in circular buffer
  imu_buffer[buffer_idx][0] = r.ax;
  imu_buffer[buffer_idx][1] = r.ay;
  imu_buffer[buffer_idx][2] = r.az;
  imu_buffer[buffer_idx][3] = r.gx;
  imu_buffer[buffer_idx][4] = r.gy;
  imu_buffer[buffer_idx][5] = r.gz;
  buffer_idx = (buffer_idx + 1) % SEQUENCE_LEN;

  // Mark ready after one full revolution
  static int sample_count = 0;
  if (!buffer_ready) {
    sample_count++;
    if (sample_count >= SEQUENCE_LEN) buffer_ready = true;
  }

  if (!buffer_ready) return;

  // ── Pre-filter: skip inference if no significant motion ──────────────────
  float mag = sqrtf(r.ax * r.ax + r.ay * r.ay + r.az * r.az);
  // Only run full LSTM when impact-level acceleration is detected
  // This saves ~12 ms of CPU every quiet second
  if (mag < IMPACT_ACCEL_G) return;

  // ── Run LSTM inference ───────────────────────────────────────────────────
  float fall_prob = run_inference();
  Serial.printf("[Infer] accel_mag=%.2fg  fall_prob=%.3f\n", mag, fall_prob);

  if (fall_prob >= FALL_THRESHOLD) {
    trigger_alert(fall_prob);
  }
}
