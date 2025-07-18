def play_sound_inProgress():
    if(config['PATH']['SOUND_PATH_INPROG'] == ""):
        print("Optional libraries not installed. Training Resuming...")
        return
    if(config['PATH']['SOUND_PATH_INPROG'] == ""):
        print("Optional libraries not installed. Training Resuming...")
        return 
    try:
        from playsound import playsound
        playsound(config['PATH']['SOUND_PATH_INPROG'])
    except ImportError:
        print('Optional libraries not installed. Training Resuming...')