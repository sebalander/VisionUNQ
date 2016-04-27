# Historial de cambios

2016/04/26
----------
Estuvimos corriendo mainYabo.py arreglando cositas de las ventanas y
viendo que errores aparecían, para tratar de arreglarlos.
- Decidimos pasar a OpenCV2.4 debido al bug #6009 (ver Fixes#5667
<https://github.com/Itseez/opencv/pull/6009/files>) que nos trae error
con flan.knnMatcher.
- De todos modos quizá sea más fácil usar un bruteforce matcher porque
son muy pocos features para matchear.

Instalando opencv2.4. Empiezo por sacar Opencv3.1

> ~/opencv/opencv-3.1.0/release$ sudo make uninstall

Parece que eso funcionó correctamente asi que ahora instalo la versión
por default en los repos de Ubuntu

> $ sudo aptitude install python2.7-opencv python-opencv
