#   edit_sound_files (14 February 2016)
#   resample, denoise, normalize, convert to mono.
#   Script modified for use in Pepper project
#
# 	Wendy Elvira-Garcia
#	wendyelviragarcia@gmail.com
#	Laboratori de Fonètica (University of Barcelona)

form Fulfilling Amper requirements
	sentence folder file with all the audio files to be modified
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


Create Strings as file list: "list", folder$+ "/*.wav"
numberOfFiles = Get number of strings

for ifile to numberOfFiles
	select Strings list
	fileName$ = Get string: ifile
	base$ = fileName$ - ".wav"
	base$ = fileName$ - ".WAV"

	# read the sound
	mysound= Read from file: folder$+"/"+ fileName$

pause

	# Noise Removal
    selectObject: mysound
    mysound = do ("Remove noise...", 0, 0, 0.025, 'clean_from', 'clean_to', 40, "Spectral subtraction")


	# Normalize intensity
	if normalize_intensity = 1
		selectObject: mysound
		Scale peak... 0.99996948
	endif

    nowarn Write to WAV file: targetFolder$+"/" + fileName$
    removeObject: mysound

endfor

	############# Clean #############
select all
Remove
echo All files processed