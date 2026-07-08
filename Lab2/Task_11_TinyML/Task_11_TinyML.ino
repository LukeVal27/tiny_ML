#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>

// ---------- Thresholds (tune these; document in lab report) ----------
const float RH_JUMP_THRESHOLD   = 2.0;    // %RH above baseline => humid_jump
const float TEMP_RISE_THRESHOLD = 0.5;    // degC above baseline => temp_rise
const float MAG_SHIFT_THRESHOLD = 15.0;   // uT deviation from baseline => mag_shift
const float LIGHT_CHANGE_FRAC   = 0.25;   // 25% relative change in clear => light change
const int   RGB_RATIO_DELTA     = 15;     // % change in a color ratio => color change

// ---------- Baseline / timing ----------
const float BASELINE_ALPHA = 0.02;        // EMA rate; small = slow-adapting baseline
const unsigned long UPDATE_MS   = 250;    // sample period
const unsigned long COOLDOWN_MS = 2000;   // debounce: min time between event reports
const unsigned long WARMUP_MS   = 4000;   // let baselines settle before detecting

// ---------- Baseline state ----------
float rhBase = 0, tempBase = 0, magBase = 0, clearBase = 0;
float rRatioBase = 0, gRatioBase = 0, bRatioBase = 0;
bool baselineInit = false;

unsigned long startMs = 0;
unsigned long lastEventMs = 0;
const char* lastEvent = "BASELINE_NORMAL";

// ---------- Helpers ----------
float ema(float base, float x) {
  return base + BASELINE_ALPHA * (x - base);
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  if (!HS300x.begin()) {
    Serial.println("Failed to initialize humidity/temperature sensor.");
    while (1);
  }
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }

  Serial.println("Environmental event monitor started");
  startMs = millis();
}

void loop() {
  // ---------- Read modalities ----------
  float rh   = HS300x.readHumidity();
  float temp = HS300x.readTemperature();

  static float mag = 0.0;
  float mx, my, mz;
  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    mag = sqrt(mx * mx + my * my + mz * mz);   // field magnitude, uT
  }

  static int r = 0, g = 0, b = 0, clearVal = 0;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clearVal);
  }

  // Normalized color ratios: isolate COLOR change from pure BRIGHTNESS change
  float rgbSum = (float)(r + g + b);
  if (rgbSum < 1.0) rgbSum = 1.0;
  float rRatio = 100.0 * r / rgbSum;
  float gRatio = 100.0 * g / rgbSum;
  float bRatio = 100.0 * b / rgbSum;

  // ---------- Initialize baselines on first pass ----------
  if (!baselineInit) {
    rhBase = rh;  tempBase = temp;  magBase = mag;  clearBase = clearVal;
    rRatioBase = rRatio;  gRatioBase = gRatio;  bRatioBase = bRatio;
    baselineInit = true;
  }

  // ---------- Deviations from baseline ----------
  float dRh   = rh   - rhBase;              // signed: breath raises RH
  float dTemp = temp - tempBase;            // signed: warm air raises T
  float dMag  = fabs(mag - magBase);        // magnitude: any disturbance
  float dClearFrac = fabs(clearVal - clearBase) / max(clearBase, 1.0f);
  float dColor = max(fabs(rRatio - rRatioBase),
                 max(fabs(gRatio - gRatioBase),
                     fabs(bRatio - bRatioBase)));

  // ---------- Binary event indicators ----------
  int humid_jump           = (dRh   > RH_JUMP_THRESHOLD)   ? 1 : 0;
  int temp_rise            = (dTemp > TEMP_RISE_THRESHOLD) ? 1 : 0;
  int mag_shift            = (dMag  > MAG_SHIFT_THRESHOLD) ? 1 : 0;
  int light_or_color_change = (dClearFrac > LIGHT_CHANGE_FRAC ||
                               dColor     > RGB_RATIO_DELTA) ? 1 : 0;

  // ---------- Rule-based event decision (priority ordered) ----------
  bool warm = (millis() - startMs) > WARMUP_MS;
  bool cooled = (millis() - lastEventMs) > COOLDOWN_MS;

  const char* label = "BASELINE_NORMAL";

  if (warm) {
    if (humid_jump || temp_rise) {
      label = "BREATH_OR_WARM_AIR_EVENT";
    } else if (mag_shift) {
      label = "MAGNETIC_DISTURBANCE_EVENT";
    } else if (light_or_color_change) {
      label = "LIGHT_OR_COLOR_CHANGE_EVENT";
    }
  }

  // ---------- Cooldown / debounce ----------
  // A new non-baseline event only latches if the cooldown has expired.
  if (strcmp(label, "BASELINE_NORMAL") != 0) {
    if (cooled) {
      lastEventMs = millis();
      lastEvent = label;
    } else {
      label = lastEvent;   // hold previous label, don't re-trigger
    }
  } else {
    lastEvent = "BASELINE_NORMAL";
  }

  // ---------- Baseline update: freeze during an active event ----------
  // Otherwise the EMA chases the disturbance and the event self-cancels.
  bool anyFlag = humid_jump || temp_rise || mag_shift || light_or_color_change;
  if (!anyFlag) {
    rhBase     = ema(rhBase, rh);
    tempBase   = ema(tempBase, temp);
    magBase    = ema(magBase, mag);
    clearBase  = ema(clearBase, clearVal);
    rRatioBase = ema(rRatioBase, rRatio);
    gRatioBase = ema(gRatioBase, gRatio);
    bRatioBase = ema(bRatioBase, bRatio);
  }

  // ---------- Serial Monitor output ----------
  Serial.print("raw,rh=");    Serial.print(rh, 2);
  Serial.print(",temp=");     Serial.print(temp, 2);
  Serial.print(",mag=");      Serial.print(mag, 2);
  Serial.print(",r=");        Serial.print(r);
  Serial.print(",g=");        Serial.print(g);
  Serial.print(",b=");        Serial.print(b);
  Serial.print(",clear=");    Serial.println(clearVal);

  Serial.print("flags,humid_jump=");        Serial.print(humid_jump);
  Serial.print(",temp_rise=");              Serial.print(temp_rise);
  Serial.print(",mag_shift=");              Serial.print(mag_shift);
  Serial.print(",light_or_color_change=");  Serial.println(light_or_color_change);

  Serial.print("event,");                   Serial.println(label);

  delay(UPDATE_MS);
}