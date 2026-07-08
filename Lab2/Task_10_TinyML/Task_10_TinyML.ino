#include <PDM.h>
#include <Arduino_APDS9960.h>
#include <Arduino_BMI270_BMM150.h>

// ---------- Thresholds (tune these; document in lab report) ----------
const int   MIC_THRESHOLD    = 50;    // mic level above this => sound = 1
const int   DARK_THRESHOLD   = 50;     // clear below this      => dark  = 1
const float MOTION_THRESHOLD = 0.05;   // |accel|-1g above this => moving = 1  (g)
const int   NEAR_THRESHOLD   = 100;    // proximity below this  => near  = 1  (0=closest, 255=far)

const unsigned long UPDATE_MS = 200;

// ---------- PDM microphone ----------
short sampleBuffer[256];
volatile int samplesRead = 0;
int micLevel = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

int readMicLevel() {
  if (samplesRead) {
    long sum = 0;
    for (int i = 0; i < samplesRead; i++) {
      sum += abs(sampleBuffer[i]);
    }
    micLevel = sum / samplesRead;
    samplesRead = 0;
  }
  return micLevel;
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(1500);

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM microphone.");
    while (1);
  }

  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  Serial.println("Smart Workspace Situation Classifier started");
}

// ---------- Loop ----------
void loop() {
  // --- Modality 1: audio activity ---
  int mic = readMicLevel();

  // --- Modality 2: ambient brightness (clear channel) ---
  static int r = 0, g = 0, b = 0, clearVal = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clearVal);
  }

  // --- Modality 3: physical motion (IMU accel magnitude deviation from 1g) ---
  static float motion = 0.0;
  float ax, ay, az;
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    float mag = sqrt(ax * ax + ay * ay + az * az);
    motion = fabs(mag - 1.0);   // ~0 when still, grows when moved
  }

  // --- Modality 4: user presence (proximity) ---
  static int prox = 255;
  if (APDS.proximityAvailable()) {
    prox = APDS.readProximity();   // 0 = closest, 255 = farthest
  }

  // ---------- Binary decisions ----------
  int sound  = (mic      > MIC_THRESHOLD)    ? 1 : 0;
  int dark   = (clearVal < DARK_THRESHOLD)   ? 1 : 0;
  int moving = (motion   > MOTION_THRESHOLD) ? 1 : 0;
  int near   = (prox     < NEAR_THRESHOLD)   ? 1 : 0;

  // ---------- Rule-based fusion ----------
  const char* label = "UNKNOWN";

  if (!sound && !dark && !moving && !near) {
    label = "QUIET_BRIGHT_STEADY_FAR";
  } else if (sound && !dark && !moving && !near) {
    label = "NOISY_BRIGHT_STEADY_FAR";
  } else if (!sound && dark && !moving && near) {
    label = "QUIET_DARK_STEADY_NEAR";
  } else if (sound && !dark && moving && near) {
    label = "NOISY_BRIGHT_MOVING_NEAR";
  }

  // ---------- Serial Monitor output ----------
  Serial.print("raw,mic=");    Serial.print(mic);
  Serial.print(",clear=");     Serial.print(clearVal);
  Serial.print(",motion=");    Serial.print(motion, 3);
  Serial.print(",prox=");      Serial.println(prox);

  Serial.print("flags,sound="); Serial.print(sound);
  Serial.print(",dark=");       Serial.print(dark);
  Serial.print(",moving=");     Serial.print(moving);
  Serial.print(",near=");       Serial.println(near);

  Serial.print("state,");       Serial.println(label);

  delay(UPDATE_MS);
}