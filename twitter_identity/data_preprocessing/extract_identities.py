"""
The code for extracting identities from a given set of tweets
"""

import os
from os.path import join
import re
from multiprocessing import Pool
from collections import Counter
import gzip

import emoji
import numpy as np
import pandas as pd
from tqdm import tqdm

from twitter_identity.utils.utils import get_weekly_bins

ALL_IDENTITIES = [
    'gender_sexuality', 'age', 'ethnicity', 'religion',
    'relationship', 'education', 'occupation', 'political',
    'personal', 'sensitive']

def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)

class IdentityExtactor:    
    def __init__(self, include_emojis=True):
        self.include_emojis=include_emojis
        self.identity2extractor = {
            'gender_sexuality':self.extract_gender_sexuality,
            'age':self.extract_age,
            'ethnicity':self.extract_ethnicity,
            'religion':self.extract_religion,
            'relationship':self.extract_relationship,
            'education':self.extract_education,
            'occupation':self.extract_occupation,
            'political':self.extract_political,
            'personal':self.extract_personal,
            'sensitive':self.extract_sensitive_personal
        }
        
    def distinct_emoji_list(self, string):
        """Resturns distinct list of emojis from a string"""
        return {x['emoji'] for x in emoji.emoji_list(string)}
    
    def split_description_to_substrings(self, text):
        """
        Function for splitting text based on Twitter delimiters
        :param text: text to be split
        :return: a list of tokens
        """
        if self.include_emojis:
            emojis = self.distinct_emoji_list(text)
            text = get_emoji_regexp().sub("|",text)#.encode("ascii","namereplace").decode()
            
        spl = [x.strip() for x in re.split(r"[\(\)|•*;~°,\n\t/]|[!…]+|[-–\/.]+ | [&+:]+ | [+] |([\/])(?=[A-Za-z ])|([.!-]{2,})| and |([#@][A-Za-z0-9_]+)",text.lower()) if (
            x and x.strip() !="" and not x.strip() in "|•&*#;~°.!…-/–")]
        return spl

    def extract_gender_sexuality(self, text):
        """Function for extracting phrases indicative of preferred gender of sexuality

        Args:
            text (_type_): Twitter bio string

        Returns:
            _type_: Formatted string listing all identities
        """
        
        results = []
        text=text.lower()

        ## step 1: entire string - pronouns ##
        # men pronouns
        reg = re.compile(r'\b(?:he|him|his)\s?(?:/|\s)?\s?(?:him|his)\b')
        res = reg.findall(text)
        for phrase in res:
            results.append(('gender_men',phrase))

        # women pronouns
        reg = re.compile(r'\b(?:she|her)\s?(?:/|\s)?\s?(?:she|hers?)\b')
        res = reg.findall(text)
        for phrase in res:
            results.append(('gender_women',phrase))

        # nonbinary pronouns
        reg = re.compile(r'\b(?:he|him|his|she|hers?|they|them|their)\s?(?:/|\s)?\s?(?:they|them|theirs?)\b')
        res = reg.findall(text)
        for phrase in res:
            results.append(('gender_nonbinary',phrase))

        ## step 2: for substring level
        
        # generate regular expressions
        reg_male = re.compile(r'\b(?:(?:grand?)?(?:father|dad|husband|son|brother)(?: of| to| and|$)|male\b)')
        reg_female = re.compile(r'\b(?:(?:grand?)?(?:mother|mom|mum|wife|daughter|sister)(?: of| to| and|$)|female\b)')
        reg_men=re.compile(r'\b(?:man|boy|guy|dude)\b')
        reg_women=re.compile(r'\b(?:girl|woman|gal)\b')
        reg_nb_sexual = re.compile(r'\b(?:bi(?:(?:-|\s)?sexual)|(?:a|pan|homo)-?sexual|gay|lesbian|queer|lgbt?\b)')
        reg_nb_gender = re.compile(r'\b(?:nb|non(?:\s|-)?binary|trans(?:gender|male|female)?|enby|queer|lgbt?\b)')
        
        words = self.split_description_to_substrings(text)
        # search each word while removing exceptions
        for word in words:
            res = reg_male.findall(word)
            for phrase in res:
                if re.findall(r'(?:son of a bitch|suga|my( \w+)? \b(?:grand?)?(?:father|dad|daddy|husband|son|male))',word):
                    continue
                else:
                    results.append(('sexuality_male',phrase))

            res = reg_female.findall(word)
            for phrase in res:
                if re.findall(r'(?:suga|my( \w+)? \b(?:grand?)?(?:mother|mom|mommy|mum|wife|daughter|sister|female))',word):
                    continue
                else:
                    results.append(('sexuality_female',phrase))

            res = reg_men.findall(word)
            for phrase in res:
                if re.findall(r'((of|my|oh|any|hey|to)( \w+){0,2} (man|boy|guy|dude)\b|man u(nited|td)?\b|boy with luv)',word):
                    continue
                elif re.findall(r'(mom|mother|mum|wife\w?|daughter|sister) (of|to)',word):
                    continue
                else:
                    results.append(('gender_men',phrase))

            res = reg_women.findall(word)
            for phrase in res:
                if re.findall(r'(of|my|oh|any|hey|to)( \w+){0,2} (girl|woman|gal)\b',word):
                    continue
                elif re.findall(r'(husband|dad(dy)?|father|son|brother) (of|to)',word):
                    continue
                else:
                    results.append(('gender_women',phrase))

            res = reg_nb_sexual.findall(word)
            for phrase in res:
                if re.findall(r'(bi(\s|-)?(vocation|ling|cycl|weekly|annual|monthly)|\brights\b)',word):
                    continue
                else:
                    results.append(('sexuality_nonbinary',phrase))

            res = reg_nb_gender.findall(word)
            for phrase in res:
                if re.findall(r'(bi(\s|-)?(vocation|ling|cycl|weekly|annual|monthly)|\brights\b)', word):
                    continue
                else:
                    results.append(('gender_nonbinary', phrase))

        if results:
            cat2phrases={}
            for cat,phrase in results:
                if cat not in cat2phrases:
                    cat2phrases[cat]=[]
                cat2phrases[cat].append(phrase)
            for cat,V in cat2phrases.items():
                cat2phrases[cat]=','.join(list(set(V)))
            return '|'.join([cat+':'+V for cat,V in cat2phrases.items()])
        else:
            return

    def extract_age(self, text):
        """Function for extracting phrases indicative of age

        Args:
            text (_type_): Twitter bio string

        Returns:
            _type_: Formatted string listing all identities
        """
        
        results = []
        text=text.lower()

        ## step 1: for entire string
        re_list = [
            re.compile(r'\b([0-9][0-9])\s?(?:(?:year|yr)s?(?:\s|-)old|y(?:\s|/)?o\b)'),
        ]
        for reg in re_list:
            res = reg.findall(text)
            if res:
                results.extend(res)

        ## step 2: for substring-level
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            if substring.isdigit():
                try:
                    num = int(substring)
                    if (num >= 13) & (num <= 70):
                        results.append(num)
                except:
                    pass

        for substring in substrings:
            for reg in re_list:
                res = reg.findall(substring)
                if res:
                    results.extend(res)
        if results:
            results = [int(x) for x in results]
            # results = [str(x) for x in results if (x>=13) and (x<=70)]
            results = list(set(results))
        if results:
            if len(results)==1: # we remove cases with more than 1 case
                age=results[0]
                if age <= 17:
                    cat= '13-17'
                elif age <= 24:
                    cat= '18-24'
                elif age <= 34:
                    cat= '25-34'
                elif age <= 49:
                    cat= '35-49'
                else:
                    cat= '50+'
                return 'age_%s:%d'%(cat,age)
            else:
                return
        else:
            return

    def extract_ethnicity(self, text):
        """Function for extracting phrases indicative of ethnicity

        Args:
            text (_type_): Twitter bio string

        Returns:
            _type_: Formatted string listing all identities
        """
        
        results = []
        re_list = [
            re.compile(
                r'\b(african?|american?|asian|british|canada|canadian|mexican|england|english|european|french|indian|irish|japanese|spanish|uk|usa)\b'
            ),
            re.compile(
                r'\b(🇺🇸|🇬🇧|🇨🇦|🇮🇪|🇧🇷|🇲🇽|󠁧󠁢󠁧🇯🇵|🇪🇸|🇮🇹|🇫🇷|🇩🇪|🇳🇱|🇮🇳|🇮🇩|🇹🇷|🇸🇦|🇹🇭|🇵🇭|🇦🇷|🇰🇷|🇪🇬|🇲🇾|🇨🇴)',
            )
        ]
        re_exclude_list = [
            re.compile(r'(learn|stud(?:y|ie)|language|lingual|speak|food|dish|cuisine|culture|music|drama)'),
            re.compile(r'\b(asian|black)\s?hate')
        ]
        for reg in re_list:
            res = reg.findall(text)
            if res:
                flag=False
                for reg in re_exclude_list:
                    if reg.search(text):
                        flag=True
                        break
                if ~flag:
                    results.extend(res)
        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return

    def extract_religion(self, text):
        results = []

        re_list=[
            re.compile(r"\b(allah|athies(?:t|m)|catholic|christ(?:ian(?:ity)?)?|church|god's|(?:of|for) god|god (?:comes )?first|god is|muslim|psalms?|philippians)\b")
            # re.compile(r"\b(islam\w*|messiah|muslim|hind(?:u|i)\w*|christ(?:ian)|church|jesus|catholic|athies(?:t|m)|(?:of|for) god|god's|god (?:comes )?first|pastor|sermon)\b"),
            # re.compile(r'((?:romans|james|proverbs|isiah|galatians|ephesians|paul|john|mark|luke|psalms|genesis|corinthians|philippians) [0-9]+\:)')
        ]
        re_exclude_list=[
            re.compile(r'(school|\bhs\b|univ|for\s?sake|god(father|mother|dess| of)|swear to god|my god)')
        ]
        for reg in re_list:
            res = reg.findall(text)
            if res:
                flag=False
                for reg in re_exclude_list:
                    if reg.search(text):
                        flag=True
                        break
                if ~flag:
                    results.extend(res)
        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return

    def extract_relationship(self, text):
        results = []
        ## step 1: for entire string
        re_list=[
            re.compile(r'\b(married|(?:father|mother|dad|daddy|mom|mum|husband|wifey?|son|daughter) (?:of|to))'),
            re.compile(r'\b(?:single|a|happy|proud|lucky|working|devoted|loving|blessed|busy|boy|girl|hockey|soccer|baseball|regular)\s?(father|mother|dad|mom|mum|husband|wife)')
        ]
        re_exclude_list=[
            re.compile(r'\b(my|your|ur) (\w+ )?(father|mother|dad|mom(ma)?|mum)\b'),
            re.compile(r'(dog|cat|pup|kitt(en|ies|y)|fur(ry|bab)|pet)'),
            re.compile(r'sugar')
        ]

        for reg in re_list:
            res = reg.findall(text)
            if res:
                flag=False
                for reg in re_exclude_list:
                    if reg.search(text):
                        flag=True
                        break
                if ~flag:
                    results.extend(res)

        ## step 2: for substring
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            if substring in set(['father','dad','daddy','mom','mommy','momma','mother','mum','husband','wife']):
                results.append(substring)
        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return

    def extract_education(self, text):
        results = []


        re_list=[
            re.compile(
                r'\b(ph(?:\.)?d|mba|student|univ(?:ersity)|college|grad(?:uate)?|college|(?:freshman|sophomore|junior|senior) at|class of|school|study(?:ing)?|\
                studie(?:s|d)|alum(?:\w+)?)\b'
            )
        ]
        re_exclude_list = [
            re.compile(r'\b(teach(?:er|ing)?|lecturer|coach(?:ing)?|prof(?:essor)?)'),
            re.compile(r'student of life')
        ]

        for reg in re_list:
            res = reg.findall(text)
            if res:
                flag=False
                for reg in re_exclude_list:
                    if reg.search(text):
                        flag=True
                        break
                if ~flag:
                    results.extend(res)

        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return

    def extract_occupation(self, text):
        results = []
        text=text.lower()
        reg_student=re.compile(r'(?:student\b|(?:junior|freshman|senior|sophomore) (?:at|@|year)|\bclass of\b|(?:grad(?:uate)?|med(?:ical)?|pharm(?:acy)?|law) school| ph/.?d candidate|college|\buniv(?:ersity)?\b)')

        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            # remove false phrases
            if re.findall(r'(future|\baspir|\bex\b|former)',substring):
                continue
            # student
            res=reg_student.findall(substring)
            if res:
                if re.findall(r'(?:life(?:long)?|(?:our|my) student|affair|director|office|teach|coach|professor|researcher|lecturer|graduated|\balum)',substring):
                    pass
                else:
                    for phrase in res:
                        results.append(('occupation_student',phrase))
            # financial
            reg_financial = re.compile(r'(?:accountant|trader|investor|banker|analyst|ceo|marketer)\b')
            res = reg_financial.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_business', phrase))

            # influencer
            reg_influencer = re.compile(r'(?:streamer|youtuber|podcaster|influencer)')
            res = reg_influencer.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_influencer', phrase))

            # healthcare
            reg_healthcare = re.compile(r'(?:dentist|doctor|nurse|physician|pharmacist|therapist|psychiatrist|dermatologist|veterinarian)\b')
            res=reg_healthcare.findall(substring)
            if res:
                for phrase in res:
                    if phrase=='doctor':
                        if re.findall(r'(doctor strange|doctor who)',text):
                            continue
                    results.append(('occupation_healthcare',phrase))

            # academia
            reg_academia = re.compile(r'\b(?:scientist|teacher|researcher|research assistant|scholar|coach|educator|instructor|lecturer|prof|professor|trainer)\b')
            res=reg_academia.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_academia',phrase))

            # art-related
            reg_art = re.compile(r'\b(?:artist|animator|creator|dancer|designer|dj|filmmaker|illustrator|musician|photographer|producer|rapper|singer|songwriter)\b')
            res=reg_art.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_art',phrase))

            # tech
            reg_art = re.compile(r'\b(?:engineer|architect|programmer|developer)\b')
            res=reg_art.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_tech',phrase))

            # news
            reg_news = re.compile(r'\b(?:journalist|reporter|correspondent)\b')
            res=reg_news.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_news',phrase))

            # writing
            reg_writing = re.compile(r'\b(?:writer|blogger|editor|author|poet|publisher)\b')
            res=reg_writing.findall(substring)
            if res:
                for phrase in res:
                    results.append(('occupation_writing',phrase))

        if results:
            cat2phrases={}
            for cat,phrase in results:
                if cat not in cat2phrases:
                    cat2phrases[cat]=[]
                cat2phrases[cat].append(phrase)
            for cat,V in cat2phrases.items():
                cat2phrases[cat]=','.join(list(set(V)))
            return '|'.join([cat+':'+V for cat,V in cat2phrases.items()])
        else:
            return

    def extract_political(self, text):
        results = []
        text=text.lower()

        # conservative pronouns
        reg_negate = re.compile(r'(?:\bnot?\b|\bnever|hat(?:e|red|ing)|\banti|\bex\b|idiot|stupid|dumb|fool|wrong|enemy|worst|dump|dislike|detest|despise|troll|impeach|imprison|fuck|danger|threat|terrible|horrible|survive|shit)')
        reg_con_list=[
            re.compile(r'\b(?:maga(?:2020)?|(?:1|2)a|kag(?:2020)?|build\s?the\s?wall|america\s?first|\bpro(?:\s|-)?life|make\s?america\s?great\s?again)\b'),
            re.compile(r'\b(?:(?:neo-?)?conservative|traditionalist|nationalist|libertarian|right(?:-|\s)?(?:ist|wing|republican))\b'),
        ]

        reg_lib_list=[
            re.compile(
                r'\b(?:liberal|progressive|socialist|democrat|left(?:\s|-)?(?:wing|ist)|blue\s?wave|equal\s?(?:ist|ism|ity|right)|marxist)\b'),
        ]
        
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            for reg in reg_con_list:
                res=reg.findall(substring)
                for phrase in res:
                    res_negate=reg_negate.findall(substring)
                    if res_negate:
                        results.append(('political_anti_conservative','_'.join(res_negate)+'_'+phrase))
                    else:
                        results.append(('political_conservative',phrase))
            for reg in reg_lib_list:
                res=reg.findall(substring)
                for phrase in res:
                    res_negate=reg_negate.findall(substring)
                    if res_negate:
                        results.append(('political_anti_liberal','_'.join(res_negate)+'_'+phrase))
                    else:
                        results.append(('political_liberal',phrase))

            reg_blm_list=[
                re.compile(r'(?:black\s?lives\s?matter|blm)')
            ]
            for reg in reg_blm_list:
                res=reg.findall(substring)
                for phrase in res:
                    results.append(('political_blm',phrase))

        if results:
            cat2phrases={}
            for cat,phrase in results:
                if cat not in cat2phrases:
                    cat2phrases[cat]=[]
                cat2phrases[cat].append(phrase)
            for cat,V in cat2phrases.items():
                cat2phrases[cat]=','.join(list(set(V)))
            return '|'.join([cat+':'+V for cat,V in cat2phrases.items()])
        else:
            return

    def extract_personal(self, text):
        results = []
        re_list = [
            re.compile(
                r'\b(insta(?:gram)?|ig|youtube|yt|cash\s?app|tumblr|twitch|venmo|web\s?(?:site|page)|whatsapp|e-?mail|snapchat|' + \
                r'onlyfans|fb|facebook|discord|tiktok|parler)\b'),
        ]
        for reg in re_list:
            res = reg.findall(text)
            if res:
                results.extend(res)

        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return

    def extract_sensitive_personal(self, text):
        results = []
        re_list = [
            re.compile(
                r'\b(surviv(?:ed|or)|depress(?:ed|ion)|autis(?:m|tic)|anxiety|adhd|diabetes|fibromyalgia|cancer|trauma(?:tized|tizing)?|' + \
                r'victim|brain injury|strokes|ptsd|chronic (?:pain|illness)|jobless|homeless|unemployed|disorder|dyslexi(?:a|c))\b')
        ]
        re_exclude_list = [
            re.compile(r'(zodiac)')
        ]

        for reg in re_list:
            res = reg.findall(text)
            if res:
                flag=False
                for reg in re_exclude_list:
                    if reg.search(text):
                        flag=True
                        break
                if ~flag:
                    results.extend(res)

        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return
        
    def extract_identities_from_profiles(self, inputs, list_of_identities=ALL_IDENTITIES):
        """From a list of user objects, extracts their identities and creates a dictionary where key: user id, value: list of identities extractes from each profile, sorted by date

        Args:
            inputs (list): A list of tuples where the values are (user id, timestamp, description)
            list_of_identities (list, optional): _description_. Defaults to ['gender_sexuality', 'age', 'ethnicity', 'religion', 'relationship', 'education', 'occupation', 'political', 'personal', 'sensitive'].
        """
        
        # Assertion on list of identities        
        assert len(list_of_identities), "list_of_identities should contain at least one identity!"
        id_list=', '.join(ALL_IDENTITIES)
        for identity in list_of_identities:
            assert identity in ALL_IDENTITIES, f"{identity} does not belong to our list of identities! It should be either {id_list}"
        
        # store all profiles by each user
        uid2profiles = {}
        for uid,ts,desc in inputs:
            if uid not in uid2profiles:
                uid2profiles[uid] = []
            uid2profiles[uid].append((float(ts), desc))
        
        # extract all required identities
        uid2profile_identities = {uid: [] for uid in uid2profiles.keys()}
        cnt = 0
        for uid, V in uid2profiles.items():
            cnt += 1
            for v in sorted(V):
                description = v[1]
                obj = {'dt': v[0]}
                
                for identity,extractor in self.identity2extractor.items():                    
                    res = extractor(description.lower())
                    if res:
                        res = '|'.join(list(set(res.split('|'))))
                        obj[identity] = res
                uid2profile_identities[uid].append(obj)
                
        return uid2profile_identities
        

    def extract_from_all_profiles(self, identity, ex):

        uid2profiles = {}
        base_dir = '/shared/2/projects/bio-change/data/user-descriptions'
        


        out = []
        for i, (uid, V) in enumerate(uid2profile_identities.items()):
            tmp_out = [uid]
            curr = 0
            phrase = ''
            for obj in V:
                dt = obj['dt']
                w1 = get_weekly_bins(dt)
                w2 = int(np.ceil(w1))
                if curr == w2:
                    continue
                for i in range(curr, w2):
                    tmp_out.append(phrase)
                #         out.append((i,phrase))
                if len(obj) == 1:
                    phrase = ''
                else:
                    phrase = obj[identity]
                    # phrase = '|'.join(obj[identity])
                curr = w2
            if curr < 57:
                for i in range(curr, 57):
                    tmp_out.append(phrase)

            out.append(tmp_out)

        df = pd.DataFrame(out, columns=['user_id'] + ['week_%02d' % i for i in range(57)])

        save_dir='/shared/2/projects/bio-change/working-dir/weekly-all-identities/1plus_changes/%s.df.tsv.gz'%identity
        df.to_csv(save_dir,sep='\t',index=False,compression='gzip')
        return


    def process_extraction(id_name,id_fn,file_name):
        load_dir='/shared/2/projects/bio-change/data/4.description-changes/'
        save_file_name = '01.%s.%s.tsv.gz'%(file_name.split('.')[0],id_name)
        save_dir='/shared/2/projects/bio-change/data/4.description-changes/preprocessing'
        uid2bio={}
        # columns=['user_id']+['week_%02d'%i for i in range(-4,60)]
        # n_weeks = len(range(-4,60))
        with gzip.open(join(load_dir,file_name)) as f:
            for line in f:
            # for line in tqdm(f,total=6073057):
                line=line.decode('utf-8')
                uid,ts,description=line.strip().split('\t')
                if uid not in uid2bio:
                    uid2bio[uid] = []
                week=get_weekly_bins(float(ts))
                if (week < -4) or (week >= 60):
                    continue
                res=id_fn(description.lower())
                uid2bio[uid].append((week,res))
                # if res:
                #     uid2bio[uid].append('week_%02d|'%week+res)
                # else:
                #     uid2bio[uid].append('week_%02d|'%week+'None')

        with gzip.open(join(save_dir,save_file_name),'wt') as outf:
            for uid,V in uid2bio.items():
                flag=False
                output=[]
                for week,res in V:
                    if res==None:
                        output.append('week_%02d|None'%week)
                    else:
                        output.append('week_%02d|'%week+res)
                        flag=True
                if flag:
                    outf.write('\t'.join([uid]+output)+'\n')
        print("Finished processing and saving ",id_name,file_name)
        return

def test_individual_extraction(text,identity='gender_sexuality'):
    IdEx = IdentityExtactor()
    return IdEx.identity2extractor[identity](text)









if __name__=='__main__':
    # test 
    text='23 | he/him | cancer'
    res=test_individual_extraction(text,'gender_sexuality')
    print(res)

