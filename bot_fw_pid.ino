#include <Robojax_L298N_DC_motor.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SH1106.h>
#include "BluetoothSerial.h"


#define TIMER_WIDTH 16

// motor 1 settings
#define CHA 0
#define ENA 4
#define IN1 16
#define IN2 17

// motor 2 settings
#define IN3 18
#define IN4 5
#define ENB 19
#define CHB 1

// Direction Encoding
const int CCW = 2;
const int CW  = 1;

// Motor Index
#define motor1 1
#define motor2 2

// Create Robot Object
Robojax_L298N_DC_motor robot(IN1, IN2, ENA, CHA,  IN3, IN4, ENB, CHB);

// OLED Display Pins
#define OLED_SDA 21
#define OLED_SCL 22
Adafruit_SH1106 display(OLED_SDA, OLED_SCL);  // create OLED Object

//Ultrasonic Sensor Pins
#define trigPin 2
#define echoPin 23

// Pan & Tilt Servo Pins
#define servoPin 14
#define servoPin2 12

// Initialize Bluetooth Communication
BluetoothSerial SerialBT;

// Declare Global Variables
int servo1_val;                       // Servo1 value buffer
int servo2_val;                       // Servo2 value buffer
int incomingByte;                    // Signal buffer
long duration;                        // Ultrasonic sensor value buffer
int thrustl;
int thrustr;
int dirl;
int dirr;
String msgl;
String msgr;
String msgp;
String msgt;

void setup() {
  Serial.begin(115200);                // Begin Serial Monitor
  robot.begin();                       // Begin robot object

  // Initialize OLED Display
  display.begin(SH1106_SWITCHCAPVCC, 0x3C);
  display.display();
  delay(2000);
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setRotation(0);
  display.clearDisplay();

  // Initialize Bluetooth
  SerialBT.begin("ESP32test");

  // Initialize the ultrasonic sensor pins
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  // Initialize Pan & Tilt Servos
  pinMode(servoPin, OUTPUT);
  pinMode(servoPin2, OUTPUT);
  ledcSetup(2, 50, TIMER_WIDTH);  // Channel 2, 50 Hz
  ledcAttachPin(servoPin, 2);
  ledcSetup(3, 50, TIMER_WIDTH);  // Channel 3, 50 Hz
  ledcAttachPin(servoPin2, 3);

  // Camera Starting Position
  look_straight();
  look_down(30);
  halt();
}

void loop() {
  if(Serial.available()) {
    incomingByte = Serial.read();
  } 
  print_heading(incomingByte);

  if(incomingByte == 0) {
    thrustl = thrustr = 80;
    dirr = dirl = CW;
  } else if (incomingByte < 0) {
    thrustl = 75;
  } else if (incomingByte > 0) {
    thrustr = 75;
  }

  //if(proximity_read() > 10) {
    robot.rotate(motor1, thrustl, dirl);
    robot.rotate(motor2, thrustr, dirr);
  //}
}

/*
    CUSTOM FUNCTIONS DEFINED BELOW
*/


///////////////////////////////
//  Gyro Print Function      //
///////////////////////////////


///////////////////////////////
//  Proximity Read Function  //
///////////////////////////////

int proximity_read() {

  // Caliberation
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);

  // Generate Pulse
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Reads the echoPin, returns the sound wave travel time in microseconds
  duration = pulseIn(echoPin, HIGH);

  // Calculating and return the distance
  return(duration * 0.034 / 2);
}

///////////////////////////////////
//  OLED Display Print Function  //
///////////////////////////////////
void print_heading(double heading) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.print("Heading: ");
  display.println(heading);
  display.setCursor(0, 20);
  display.print("Proximity: ");
  display.println(proximity_read());
  display.display();
}

void print_oled(String q1, int thrustl, String msgl, String q2, int thrustr, String msgr, String degthrust) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 0);
  display.print(q1);
  display.print(":");
  display.setCursor(25, 0);
  display.print(msgl);
  display.setCursor(63, 0);
  display.print(q2);
  display.print(":");
  display.setCursor(94, 0);
  display.print(msgr);
  display.setCursor(0, 13);
  display.print(degthrust);
  display.print(":");
  display.setCursor(25, 13);
  display.print(thrustl);
  display.setCursor(94, 13);
  display.print(thrustr);
}

void print_prox() {
  display.println(" ");
  display.println(" ");
  display.println(" ");
  display.print("Prox: ");
  display.print(proximity_read());
  display.print(" cm");
}

////////////////////////////////
//  Cam Adjustment Functions  //
////////////////////////////////

void look_down(int d) {                                                       
  servo1_val += d;
  if(180-servo1_val > 90) {
    msgp = "Up";
  } else if (180-servo1_val == 90) {
    msgp = "Str.";
  } else {
    msgp = "Down";
  }
  print_oled("Pan", (180-servo1_val), msgp, "Tilt", servo2_val, msgt, "Deg");
  print_prox();
  display.display();
  uint32_t duty = (((servo1_val/180.0)*2000)/20000.0*65536.0) + 1634;         
  ledcWrite(2, duty);                                                         
}

void look_up(int d) {
  servo1_val -= d;
  if(180-servo1_val > 90) {
    msgp = "Up";
  } else if (180-servo1_val == 90) {
    msgp = "Str.";
  } else {
    msgp = "Down";
  }
  print_oled("Pan", (180-servo1_val), msgp, "Tilt", servo2_val, msgt, "Deg");
  print_prox();
  display.display();
  uint32_t duty = (((servo1_val/180.0)*2000)/20000.0*65536.0) + 1634;
  ledcWrite(2, duty);
}

void look_right(int d) {
  servo2_val -= d;
  if(180-servo2_val > 90) {
    msgt = "Right";
  } else if (180-servo2_val == 90) {
    msgt = "Str.";
  } else {
    msgt = "Left";
  }
  print_oled("Pan", (180-servo1_val), msgp, "Tilt", servo2_val, msgt, "Deg");
  print_prox();
  display.display();
  uint32_t duty = (((servo2_val/180.0)*2000)/20000.0*65536.0) + 1634;
  ledcWrite(3, duty);
}

void look_left(int d) {
  servo2_val += d;
  if(180-servo2_val > 90) {
    msgt = "Right";
  } else if (180-servo2_val == 90) {
    msgt = "Str.";
  } else {
    msgt = "Left";
  }
  print_oled("Pan", (180-servo1_val), msgp, "Tilt", servo2_val, msgt, "Deg");
  print_prox();
  display.display();
  uint32_t duty = (((servo2_val/180.0)*2000)/20000.0*65536.0) + 1634;
  ledcWrite(3, duty);
}

void look_straight() {
  servo1_val = 90;
  servo2_val = 90;
  msgp = "Str.";
  msgt = "Str.";
  print_oled("Pan", (180-servo1_val), msgp, "Tilt", servo2_val, msgt, "Deg");
  print_prox();
  display.display();
  uint32_t duty = (((servo1_val/180.0)*2000)/20000.0*65536.0) + 1634;
  ledcWrite(2, duty);
  delay(500);
  ledcWrite(3, duty);
}

////////////////////////////////////
//  Cam Adjustment Decision Tree  //
////////////////////////////////////

void cam_control(char incomingByte) {
  if(incomingByte == 'u') { look_up(10);}
  if(incomingByte == 'j') { look_down(10);}
  if(incomingByte == 'l') { look_left(10);}
  if(incomingByte == 'r') { look_right(10);}
  if(incomingByte == 'g') { look_straight();}
}

/////////////////////////////////
//  Robot Thruster Functions   //
/////////////////////////////////

void halt() {
  thrustl = 0;
  thrustr = 0;
  /*msgl = "HALT";
  msgr = "HALT";
  print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
  print_prox();
  display.display();*/
  robot.brake(1);
  robot.brake(2);
}

void turn_left() {
  if(thrustl > 70 && thrustr > 70) {
    thrustl -= 5;
    msgl = "F";
    print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
    print_prox();
    display.display();
    robot.rotate(motor1, thrustl, dirl);
  } else {
    thrustr += 80;
    dirr = CW;
    msgr = "F";
    print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
    print_prox();
    display.display();
    robot.rotate(motor2, thrustr, dirr); 
  }
}

void turn_right() {
  if(thrustr > 70 && thrustl > 70) {
      thrustr -= 5;
      msgr = "F";
      print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
      print_prox();
      display.display();
      robot.rotate(motor2, thrustr, dirr);
  } else {
      thrustl += 80;
      dirl = CW;
      msgl = "F";
      print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
      print_prox();
      display.display();
      robot.rotate(motor1, thrustl, dirl); 
  }
}

void reverse() {
  if(dirl != CCW || dirr != CCW) {
    dirl = CCW;
    dirr = CCW;
    halt();
  }
  thrustl = 80;
  thrustr = 80;
  msgl = "R";
  msgr = "R";
  print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
  print_prox();
  display.display();
  robot.rotate(motor1, thrustl, dirl);
  robot.rotate(motor2, thrustr, dirr);
}

void forward() {
    if(dirl != CW || dirr != CW) {
      dirl = CW;
      dirr = CW;
      halt();
      delay(200);
    }
    if(proximity_read() > 10) {
      thrustl = 80;
      thrustr = 80;
      msgl = "F";
      msgr = "F";
      print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
      print_prox();
      display.display();
      robot.rotate(motor1, thrustl, dirl);
      robot.rotate(motor2, thrustr, dirr);
    } else {
      halt();
    }
}

void acc_forward(int t) {
  dirl = CCW;
  dirr = CCW;
  msgl = "F";
  msgr = "F";
  for(int i = 50; i <= 100; i++) {
    thrustl = i;
    thrustr = i;
    print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
    print_prox();
    display.display();
    robot.rotate(motor1, thrustl, dirl);
    robot.rotate(motor2, thrustr, dirr);
    delay(100);
  }
  delay(t);
  halt();
}

void acc_reverse(int t) {
  dirl = CW;
  dirr = CW;
  msgl = "R";
  msgr = "R";
  for(int i = 50; i <= 100; i++) {
    thrustl = i;
    thrustr = i;
    print_oled("L", thrustl, msgl, "R", thrustr, msgr, "Thr:");
    print_prox();
    display.display();
    robot.rotate(motor1, thrustl, dirl);
    robot.rotate(motor2, thrustr, dirr);
    delay(100);
  }
  delay(t);  
  halt();
}

////////////////////////////////////
//  Robot Thruster Decision Tree  //
////////////////////////////////////
