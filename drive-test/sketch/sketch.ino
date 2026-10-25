#include "Arduino_RouterBridge.h"

const int M1A = 5;   // left motor
const int M1B = 3;
const int M2A = 10;  // right motor
const int M2B = 9;
const int TRIG = 11;
const int ECHO = 12;

const unsigned long WATCHDOG_MS = 1000;   // stop motors if no drive() call for this long

volatile unsigned long lastDrive = 0;     // 0 means motors are already stopped

void setMotors(int left, int right) {
    if (left >= 0) { analogWrite(M1A, left);  analogWrite(M1B, 0); }
    else           { analogWrite(M1A, 0);     analogWrite(M1B, -left); }

    if (right >= 0) { analogWrite(M2A, right); analogWrite(M2B, 0); }
    else            { analogWrite(M2A, 0);     analogWrite(M2B, -right); }
}

// left/right: -255 to 255. Positive = forward, negative = backward.
void drive(int left, int right) {
    setMotors(left, right);
    lastDrive = (left == 0 && right == 0) ? 0 : millis();
}

// Returns distance in mm, or -1 if no echo within the timeout.
int distance_mm() {
    digitalWrite(TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG, LOW);
    unsigned long us = pulseIn(ECHO, HIGH, 15000);   // 15 ms timeout, about 2.5 m
    if (us == 0) return -1;
    return (int)(us * 0.343 / 2);
}

void setup() {
    pinMode(M1A, OUTPUT);
    pinMode(M1B, OUTPUT);
    pinMode(M2A, OUTPUT);
    pinMode(M2B, OUTPUT);
    pinMode(TRIG, OUTPUT);
    pinMode(ECHO, INPUT);

    setMotors(0, 0);

    Bridge.begin();
    Bridge.provide("drive", drive);
    Bridge.provide("distance_mm", distance_mm);
}

void loop() {
    unsigned long t = lastDrive;
    if (t != 0 && millis() - t > WATCHDOG_MS) {
        setMotors(0, 0);
        lastDrive = 0;
    }
}