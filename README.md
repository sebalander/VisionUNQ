# VisionUNQ
Python algorithm for traffic videos processing in VCA and regular cameras. BG/FG combined with SURF features to measure velocity. Camera georeferencing and mapping pixel coordinates in the video to coordinates in a map. Camera (VCA and PTZ) comunications with Onvif. This proyect is still in development, not organized or fully working yet.

We chose the same license as OpenCV: BSD 3-Clause License.

Group members: Agustín Yabo, Damián Estanganelli, Damián Oliva, Lilian García y Sebastián Arroyo.

Use
---
To use this python code there is no installation, just copy the folder visionUNQ to your computer and import it as a package. Runs on python2 (intended to run with python3) and OpenCV-3.1.0.

Requires: onvif (installed from source as root?), wxPython, pickle, PyQt.

Repository Content
------------------
- visionUNQ : is a packaje containing functions, objects, etc.
- examples : has working scripts accompanied by video files if necesary.
- dev : contains scrips and pices of code in development/test also mains that run using visionUNQ but haven't yet been fully tested of given definite form.
- resources : contains all non code files (images, videos, etc).
- README.md : this file
- LICENSE.md : The license that applies to all the code in the repository. BSD 3-Clause license, same as OpenCV.

Project Description
-------------------
This repository is meant to unify the python codes of people working in computer vision applied to traffic monitoring in Universidad Nacional de Quilmes (Buenos Aires, Argentina), Departamento de Ciencia y Tecnología, IACI. The idea is to organize the algorithms into a single library for easier collaboration.


Asociated Articles
------------------
Stanganelli, D., Oliva, D. E., Noblia, M., & Safar, F. (2014, June). Calibración de una cámara fisheye comercial con el modelo unificado para la observación de objetos múltiples. In Biennial Congress of Argentina (ARGENCON), 2014 IEEE (pp. 147-152). IEEE. 

Procedeeng de la 2015 XVI Workshop on Information Processing and Control (RPIC) en IEEE Xplore

Tesis Yabo

Tesis Stanganelli (proximamente)
