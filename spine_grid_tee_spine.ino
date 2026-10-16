#include <Wire.h>

// Existing pin assignments — unchanged
const int SDA_HUB = 8;
const int SCL_HUB = 9;

const int SDA_SH = 4;
const int SCL_SH = 5;

const int FLEX1 = 2;
const int FLEX2 = 1;

const uint8_t MPU_UPPER = 0x68;
const uint8_t MPU_LEFT  = 0x69;
const uint8_t MPU_RIGHT = 0x68;

const uint32_t SAMPLE_INTERVAL_US = 40000; // 25 Hz

bool upperOK = false;
bool leftOK = false;
bool rightOK = false;

struct IMUData {
  float ax, ay, az;
  float gx, gy, gz;
};

bool checkMPU(TwoWire &bus, uint8_t address) {
  bus.beginTransmission(address);
  return bus.endTransmission() == 0;
}

void writeRegister(TwoWire &bus, uint8_t address,
                   uint8_t reg, uint8_t value) {
  bus.beginTransmission(address);
  bus.write(reg);
  bus.write(value);
  bus.endTransmission();
}

void initMPU(TwoWire &bus, uint8_t address) {
  // Wake sensor
  writeRegister(bus, address, 0x6B, 0x00);

  // Accelerometer range: +/- 2 g
  writeRegister(bus, address, 0x1C, 0x00);

  // Gyroscope range: +/- 250 degrees/second
  writeRegister(bus, address, 0x1B, 0x00);

  // Sample-rate divider: 0 (internal default rate)
  writeRegister(bus, address, 0x19, 0x00);

  delay(50);
}

int16_t readInt16(TwoWire &bus) {
  uint16_t value = ((uint16_t)bus.read() << 8)
                 | (uint16_t)bus.read();
  return (int16_t)value;
}

bool readMPU(TwoWire &bus, uint8_t address, IMUData &d) {
  // Register 0x3B: accel XYZ, temperature, gyro XYZ
  bus.beginTransmission(address);
  bus.write(0x3B);

  if (bus.endTransmission(false) != 0) return false;

  if (bus.requestFrom(address, (uint8_t)14) != 14) return false;

  int16_t ax = readInt16(bus);
  int16_t ay = readInt16(bus);
  int16_t az = readInt16(bus);
  readInt16(bus); // Ignore temperature
  int16_t gx = readInt16(bus);
  int16_t gy = readInt16(bus);
  int16_t gz = readInt16(bus);

  // Convert to the units used by the repository's raw sensor pipeline
  d.ax = ax / 16384.0f;
  d.ay = ay / 16384.0f;
  d.az = az / 16384.0f;

  d.gx = gx / 131.0f;
  d.gy = gy / 131.0f;
  d.gz = gz / 131.0f;

  return true;
}

void printIMU(TwoWire &bus, uint8_t address, bool sensorOK) {
  IMUData d;

  if (!sensorOK || !readMPU(bus, address, d)) {
    Serial.print("null");
    return;
  }

  // [ax, ay, az, gx, gy, gz]
  Serial.printf(
    "[%.5f,%.5f,%.5f,%.3f,%.3f,%.3f]",
    d.ax, d.ay, d.az, d.gx, d.gy, d.gz
  );
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  analogReadResolution(12);

  Wire.begin(SDA_HUB, SCL_HUB, 100000);
  Wire1.begin(SDA_SH, SCL_SH, 100000);

  upperOK = checkMPU(Wire, MPU_UPPER);
  leftOK  = checkMPU(Wire1, MPU_LEFT);
  rightOK = checkMPU(Wire1, MPU_RIGHT);

  if (upperOK) initMPU(Wire, MPU_UPPER);
  if (leftOK)  initMPU(Wire1, MPU_LEFT);
  if (rightOK) initMPU(Wire1, MPU_RIGHT);

  Serial.println("Spine-Grid live IMU stream");
  Serial.printf(
    "Sensors: upper=%s left=%s right=%s\n",
    upperOK ? "OK" : "MISSING",
    leftOK ? "OK" : "MISSING",
    rightOK ? "OK" : "MISSING"
  );

  Serial.println("Each IMU: [ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps]");
}

void loop() {
  static uint32_t nextSample = micros();
  uint32_t now = micros();

  if ((int32_t)(now - nextSample) < 0) return;
  nextSample += SAMPLE_INTERVAL_US;

  int flex1Value = analogRead(FLEX1);
  int flex2Value = analogRead(FLEX2);

  Serial.print("{\"upper\":");
  printIMU(Wire, MPU_UPPER, upperOK);

  Serial.print(",\"left\":");
  printIMU(Wire1, MPU_LEFT, leftOK);

  Serial.print(",\"right\":");
  printIMU(Wire1, MPU_RIGHT, rightOK);

  Serial.printf(
    ",\"flex1\":%d,\"flex2\":%d}\n",
    flex1Value, flex2Value
  );
}