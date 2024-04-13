import sys
import pylangacq

reader = pylangacq.Reader.from_zip(sys.argv[1])
current_speaker = ''
u = ''
turn = -1
for utterance in reader.utterances():
    speaker = list(utterance.tiers.keys())[0]
    s = utterance.tiers[speaker]
    s = s.replace('(','').replace(')','')
    s = s.replace(' .','.').replace(' ?','?')

    #Init turns
    if turn == -1:
        turn = 0
        current_speaker = speaker
        u = '<u speaker='+speaker+'>'
    if speaker != current_speaker:
        if u != '':
            u+='</u>'
        if turn == 1:
            print(u)
            turn = 0
            u = '<u speaker='+speaker+'>'
        else:
            turn = 1
            u += '<u speaker='+speaker+'>'
    u+=s+' '
    current_speaker = speaker
