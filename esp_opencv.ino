#include <esp32cam.h>
#include <WebServer.h>

WebServer server(80);         // create web server object on port #80

// Define the camera image dimensions
static auto loRes = esp32cam::Resolution::find(320, 240);
static auto hiRes = esp32cam::Resolution::find(800, 600);

const char *ssid = "Hasenhoeft";
const char *key = "21162010819665839679";

void setup() {
  Serial.begin(115200);           // begin serial monitor

  // Configure OV2640 Camera
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(hiRes);
    cfg.setBufferCount(2);
    cfg.setJpeg(80);

    bool ok = Camera.begin(cfg);
    Serial.println(ok ? "CAMERA OK" : "CAMERA FAIL");
  }
  WiFi.mode(WIFI_STA);
  Serial.print("Attempting connection to ");
  Serial.println(ssid);
  Serial.println(".");
  Serial.println(".");
  WiFi.begin(ssid, key);
  // wait 10 seconds for connection:
  delay(10000);
  if (WiFi.status() != WL_CONNECTED) {
    Serial.print(ssid);
    Serial.println(" Unavailable");
    Serial.println(".");
    Serial.println(".");
    Serial.println("Starting Virtual Access Point...");
    // Configure module to Access Point mode
    WiFi.mode(WIFI_AP);
    WiFi.softAP("bot_cam");
    delay(1500);
    // Print IP Address to Serial Monitor
    Serial.print("Access Camera at: ");
    Serial.print("http://");
    Serial.println(WiFi.softAPIP());
    Serial.println("  /cam.bmp");
    Serial.println("  /cam-lo.jpg");
    Serial.println("  /cam-hi.jpg");
    Serial.println("  /cam.mjpeg");
  } else {
    // Print IP Address to Serial Monitor
    Serial.print("http://");
    Serial.println(WiFi.localIP());
    Serial.println("  /cam.bmp");
    Serial.println("  /cam-lo.jpg");
    Serial.println("  /cam-hi.jpg");
    Serial.println("  /cam.mjpeg");
  }

  server.on("/cam.bmp", handleBmp);
  server.on("/cam-lo.jpg", handleJpgLo);
  server.on("/cam-hi.jpg", handleJpgHi);
  server.on("/cam.jpg", handleJpg);
  server.on("/cam.mjpeg", handleMjpeg);

  server.begin();
}

void loop() {
  server.handleClient();
}

////////////////////////////
//    Custom Functions    //
////////////////////////////

void handleBmp()
{
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }

  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", 
                frame->getWidth(), 
                frame->getHeight(),
                static_cast<int>(frame->size())
  );

  if (!frame->toBmp()) {
    Serial.println("CONVERT FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CONVERT OK %dx%d %db\n", 
                frame->getWidth(), 
                frame->getHeight(),
                static_cast<int>(frame->size())
  );

  server.setContentLength(frame->size());
  server.send(200, "image/bmp");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

void serveJpg() {
  auto frame = esp32cam::capture();
  if (frame == nullptr) {
    Serial.println("CAPTURE FAIL");
    server.send(503, "", "");
    return;
  }
  Serial.printf("CAPTURE OK %dx%d %db\n", 
                frame->getWidth(), 
                frame->getHeight(),
                static_cast<int>(frame->size())
  );

  server.setContentLength(frame->size());
  server.send(200, "image/jpeg");
  WiFiClient client = server.client();
  frame->writeTo(client);
}

void handleJpgLo() {
  if (!esp32cam::Camera.changeResolution(loRes)) {
    Serial.println("SET-LO-RES FAIL");
  }
  serveJpg();
}

void handleJpgHi() {
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }
  serveJpg();
}

void handleJpg() {
  server.sendHeader("Location", "/cam-hi.jpg");
  server.send(302, "", "");
}

void handleMjpeg() {
  if (!esp32cam::Camera.changeResolution(hiRes)) {
    Serial.println("SET-HI-RES FAIL");
  }

  Serial.println("STREAM BEGIN");
  WiFiClient client = server.client();
  auto startTime = millis();
  int res = esp32cam::Camera.streamMjpeg(client);
  if (res <= 0) {
    Serial.printf("STREAM ERROR %d\n", res);
    return;
  }
  auto duration = millis() - startTime;
  Serial.printf("STREAM END %dfrm %0.2ffps\n", res, 1000.0 * res / duration);
}
