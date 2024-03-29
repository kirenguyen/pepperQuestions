###########################################################################
# 	Wendy Elvira-Garcia                                                   #
#	wendyelviragarcia@gmail.com                                           #
#	Laboratori de Fonètica (University of Barcelona)                      #
###########################################################################
#                                                                         #                                                                        #
#    Free use under GNU license: see http://www.gnu.org/licenses/         #
#    Modified for use by Tran Nguyen for Yoshimoto Robotics Laboratory    #
#                                                                         #
###########################################################################

form Cleaning Noise requirements
	sentence fileName path to the singular .wav file to be modified
	sentence targetFolder where new file will be written
	boolean Rewrite 1
#	1 = Yes
#	2 = No, save files with another name
    real clean_from 80Hz
    real clean_to 10000Hz
	boolean Normalize_intensity 1
	boolean Remove_noise 1

endform


########################################


base$ = fileName$ - ".wav"
base$ = fileName$ - ".WAV"

# read the sound
mysound= Read from file: fileName$
sound_name$ = selected$("Sound")

pause

# Noise Removal
selectObject: mysound
mysound = do ("Remove noise...", 0, 0, 0.025, 'clean_from', 'clean_to', 40, "Spectral subtraction")


# Normalize intensity
if normalize_intensity = 1
    selectObject: mysound
    Scale peak... 0.99996948
endif

nowarn Write to WAV file: targetFolder$+"/" + sound_name$ + ".wav"
removeObject: mysound


############# Clean #############
select all
Remove
echo All files processed