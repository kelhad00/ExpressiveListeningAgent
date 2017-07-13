<a id="top"/> 
# squirrel_ser
This folder has source codes for an audio/visual emotion recognition system. Please fine a training script in a folder: "./scripts/".
More details will be updated soon.

Maintainer: [**batikim09**](https://github.com/**github-user**/) (**batikim09**) - **j.kim@utwente.nl**

##Contents
1. <a href="#1--installation-requirements">Installation Requirements</a>

2. <a href="#2--build">Build</a>

3. <a href="#3--usage">Usage</a>

4. <a href="#4--references">References</a>

## 1. Installation Requirements <a id="1--installation-requirements"/>
####Debian packages

Please run the following steps BEFORE you run catkin_make.

`sudo apt-get install python-pip python-dev libhdf5-dev portaudio19-dev'

Next, using pip, install all pre-required modules.
(pip version 8.1 is required.)

http://askubuntu.com/questions/712339/how-to-upgrade-pip-to-latest

If you have old numpy (<1.12) please remove it.
https://github.com/tensorflow/tensorflow/issues/559

Then,
sudo pip install -r requirements.txt

If you have numpy already, it must be higher than 1.12.0
try to install tensorflow using pip, then it will install the proper version of numpy.

## 4. References <a id="4--references"/>
