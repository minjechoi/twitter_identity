"""
The code for extracting identities from a given set of tweets
"""
import argparse
import os
from os.path import join
import re
from multiprocessing import Pool
from collections import Counter
import gzip

# import emoji
import numpy as np
import pandas as pd
from tqdm import tqdm

# from twitter_identity.utils.utils import get_weekly_bins,write_data_file_info

ALL_IDENTITIES = [
    'gender', 'age', 'ethnicity', 'religion',
    'relationship', 'education', 'occupation', 'political',
    'social_media', 'sensitive']

def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)

class IdentityExtactor:    
    def __init__(self, include_emojis=False):
        self.include_emojis=include_emojis
        self.identity2extractor = {
            'gender':self.extract_gender,
            'age':self.extract_age,
            # 'ethnicity':self.extract_ethnicity,
            'religion':self.extract_religion,
            'relationship':self.extract_relationship,
            'education':self.extract_education,
            'occupation':self.extract_occupation,
            'political':self.extract_political,
            'social_media':self.extract_social_media,
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

    def extract_gender(self, text):
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
        
        # # generate regular expressions
        # reg_male = re.compile(r'\b(?:(?:grand?)?(?:father|dad|husband|son|brother)(?: of| to| and|$)|male\b)')
        # reg_female = re.compile(r'\b(?:(?:grand?)?(?:mother|mom|mum|wife|daughter|sister)(?: of| to| and|$)|female\b)')
        # reg_men=re.compile(r'\b(?:man|boy|guy|dude)\b')
        # reg_women=re.compile(r'\b(?:girl|woman|gal)\b')
        # reg_nb_sexual = re.compile(r'\b(?:bi(?:(?:-|\s)?sexual)|(?:a|pan|homo)-?sexual|gay|lesbian|queer|lgbt?\b)')
        # reg_nb_gender = re.compile(r'\b(?:nb|non(?:\s|-)?binary|trans(?:gender|male|female)?|enby|queer|lgbt?\b)')
        
        # words = self.split_description_to_substrings(text)
        # # search each word while removing exceptions
        # for word in words:
        #     res = reg_male.findall(word)
        #     for phrase in res:
        #         if re.findall(r'(?:son of a bitch|suga|my( \w+)? \b(?:grand?)?(?:father|dad|daddy|husband|son|male))',word):
        #             continue
        #         else:
        #             results.append(('sexuality_male',phrase))

        #     res = reg_female.findall(word)
        #     for phrase in res:
        #         if re.findall(r'(?:suga|my( \w+)? \b(?:grand?)?(?:mother|mom|mommy|mum|wife|daughter|sister|female))',word):
        #             continue
        #         else:
        #             results.append(('sexuality_female',phrase))

        #     res = reg_men.findall(word)
        #     for phrase in res:
        #         if re.findall(r'((of|my|oh|any|hey|to)( \w+){0,2} (man|boy|guy|dude)\b|man u(nited|td)?\b|boy with luv)',word):
        #             continue
        #         elif re.findall(r'(mom|mother|mum|wife\w?|daughter|sister) (of|to)',word):
        #             continue
        #         else:
        #             results.append(('gender_men',phrase))

        #     res = reg_women.findall(word)
        #     for phrase in res:
        #         if re.findall(r'(of|my|oh|any|hey|to)( \w+){0,2} (girl|woman|gal)\b',word):
        #             continue
        #         elif re.findall(r'(husband|dad(dy)?|father|son|brother) (of|to)',word):
        #             continue
        #         else:
        #             results.append(('gender_women',phrase))

        #     res = reg_nb_sexual.findall(word)
        #     for phrase in res:
        #         if re.findall(r'(bi(\s|-)?(vocation|ling|cycl|weekly|annual|monthly)|\brights\b)',word):
        #             continue
        #         else:
        #             results.append(('sexuality_nonbinary',phrase))

        #     res = reg_nb_gender.findall(word)
        #     for phrase in res:
        #         if re.findall(r'(bi(\s|-)?(vocation|ling|cycl|weekly|annual|monthly)|\brights\b)', word):
        #             continue
        #         else:
        #             results.append(('gender_nonbinary', phrase))

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
                
        # remove 01/28 -like strings
        text = re.sub(r'[0-9]{1,4}(\/[0-9]{1,4}){1,}', '', text)

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
        text=text.lower()
        
        re_list = [
            re.compile(
                r'\b(african|asian|hispanic|latin(?:a|o))'
                # r'\b(african?|american?|asian|british|canada|canadian|mexican|england|english|european|french|indian|irish|japanese|spanish|uk|usa)\b'
            ),
            # re.compile(
            #     r'\b(🇺🇸|🇬🇧|🇨🇦|🇮🇪|🇧🇷|🇲🇽|󠁧󠁢󠁧🇯🇵|🇪🇸|🇮🇹|🇫🇷|🇩🇪|🇳🇱|🇮🇳|🇮🇩|🇹🇷|🇸🇦|🇹🇭|🇵🇭|🇦🇷|🇰🇷|🇪🇬|🇲🇾|🇨🇴)',
            # )
        ]
        
        reg = re.compile(r'\b(african|asian|hispanic|latin(?:a|o))')
        re_exclude_list = [
            re.compile(r'(learn|stud(?:y|ie)|language|lingual|speak|food|dish|cuisine|culture|music|drama|tv|movie)'),
            re.compile(r'\b(asian|black)\s?hate')
        ]
        # for reg in re_list:
        #     res = reg.findall(text)
        #     if res:
        #         flag=False
        #         for reg in re_exclude_list:
        #             if reg.search(text):
        #                 flag=True
        #                 break
        #         if flag==False:
        #             results.extend(res)
        
        ## step 2: for substring-level
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            res=reg.findall(substring)
            if res:
                flag=False
                for reg_exclude in re_exclude_list:
                    if reg_exclude.search(substring):
                        flag=True
                        break
                if flag==False:
                    results.extend(res)
        if results:
            results = list(set(results))
            return '|'.join([x.strip() for x in results])
        else:
            return

    def extract_religion(self, text):
        text=text.lower()
        results = []

        re_cat = re.compile(r"\b(jesus|bible|catholic|christ(?:ian(?:ity)?)?|church|psalms?|philippians|romans|proverbs)\b")
        re_mus = re.compile(r"\b(allah|muslim|islam(?:ic)?|quran|koran|hadith|prophet/s?muhammad)\b")
        re_ath = re.compile(r"\b(atheis(?:t|m))\b")
        re_hin = re.compile(r"\b(hind(?:i|u(?:ism)?))\b")
        re_gen = re.compile(r"\b(god's|(?:of|for) god|god (?:comes )?first|god is)\b")
        # re_list=[
        #     re.compile(r"\b(allah|athies(?:t|m)|catholic|christ(?:ian(?:ity)?)?|church|god's|(?:of|for) god|god (?:comes )?first|god is|muslim|psalms?|philippians)\b")
        #     # re.compile(r"\b(islam\w*|messiah|muslim|hind(?:u|i)\w*|christ(?:ian)|church|jesus|catholic|athies(?:t|m)|(?:of|for) god|god's|god (?:comes )?first|pastor|sermon)\b"),
        #     # re.compile(r'((?:romans|james|proverbs|isiah|galatians|ephesians|paul|john|mark|luke|psalms|genesis|corinthians|philippians) [0-9]+\:)')
        # ]
        re_exclude_list=[
            re.compile(r'(school|\bhs\b|univ|for\s?sake|god(father|mother|dess| of)|swear to god|my god)')
        ]

        for reg,subcategory in zip(
            [re_cat,re_mus,re_hin,re_ath,re_gen],
            ['religion_cath/christ','religion_islam','religion_hinduism','religion_atheism','religion_general']):
            res = reg.findall(text)
            if res:
                flag=False
                for reg in re_exclude_list:
                    if reg.search(text):
                        flag=True
                        break
                if flag==False:
                    results.append((subcategory,res))
        if results:
            out=[]
            for subcategory,V in results:
                V=list(set(V))
                out.append(f'{subcategory}:%s'%','.join(V))
            return '|'.join(out)
        else:
            return

    def extract_relationship(self, text):
        text=text.lower()
        results = []
        
        # ## step 1: for entire string
        # re_list=[
        #     re.compile(r'\b(?:married|(father|mother|dad|daddy|mom|mum|husband|wifey?|son|daughter) (?:of|to))'),
        #     re.compile(r'\b(?:single|a|happy|proud|lucky|working|devoted|loving|blessed|busy|boy|girl|hockey|soccer|baseball|regular)\s?(father|mother|dad|mom|mum|husband|wife)')
        # ]
        re_exclude_list=[
            re.compile(r'\b(my|y?our|ur|his|her|their) (\w+ )?(father|mother|dad|mom(ma)?|mum|sister|brother|husband|wifey?)\b'),
            re.compile(r'\b(dog|cat|pup|kitt(en|ies|y)|fur(ry|bab)|pet)s?'),
            re.compile(r'sugar')
        ]

        # for reg in re_list:
        #     res = reg.findall(text)
        #     if res:
        #         flag=False

        ## step 2: for substring
        # reg = re.compile(r'\b(?:grand?)?(father|dad|daddy|mom|mommy|momma|mother|mum|grandma|gran|granny|husband|wife|sister|brother|married|fianc(?:é|e|ée|ee))(?:$| (?:of|to))')
        reg_parent = re.compile(r'\b(?:grand?)?(father|dad|daddy|mom|mommy|momma|mother|mum|grandma|gran|granny)(?:$| (?:of|to))')
        reg_partner = re.compile(r'\b(husband|wife|married|fianc(?:é|e|ée|ee))(?:$| (?:of|to))')
        reg_sibling = re.compile(r'\b(bro(?:ther)?|sis(?:ter)?)(?:$| (?:of|to))')
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            for subcategory,reg in zip(
                ['relationship_parent','relationship_partner','relationship_sibling'],
                [reg_parent,reg_partner,reg_sibling]
            ):
                res=reg.findall(substring)
                if res:
                    flag=False
                    for reg_exclude in re_exclude_list:
                        if reg_exclude.search(substring):
                            flag=True
                            break
                    if flag==False:
                        results.append((subcategory,res))
                        
        if results:
            out=[]
            for subcategory,V in results:
                V=list(set(V))
                out.append(f'{subcategory}:%s'%','.join(V))
            return '|'.join(out)
        else:
            return
    
    

    def extract_education(self, text):
        text=text.lower()
        results = []

        reg = re.compile(
                r'\b(ph(?:\.)?d|post(?:-|\s)?doc|mba|student|univ(?:ersity)|college|grad(?:uate)?|college|(?:freshman|sophomore|junior|senior) at|class of|school|study(?:ing)?|\
                studie(?:s|d)|alum(?:\w+)?)\b'
            )

        re_exclude_list = [
            re.compile(r'\b(teach(?:er|ing)?|lecturer|coach(?:ing)?|prof(?:essor)?|principal|aspiring|future)'),
            re.compile(r'student of life')
        ]

        # for reg in re_list:
        #     res = reg.findall(text)
        #     if res:
        #         flag=False
        #         for reg in re_exclude_list:
        #             if reg.search(text):
        #                 flag=True
        #                 break
        #         if flag==False:
        #             results.extend(res)

        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            res=reg.findall(substring)
            if res:
                flag=False
                for reg_exclude in re_exclude_list:
                    if reg_exclude.search(substring):
                        flag=True
                        break
                if flag==False:
                    results.extend(res)
        if results:
            results = list(set(results))
            return 'education:%s'%'|'.join([x.strip() for x in results])
        else:
            return

    def extract_occupation(self, text):
        text=text.lower()
        results = []
        
        text=text.lower()

        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            # remove false phrases
            if re.findall(r'(future|\baspir|\bex(-|\s)|former)',substring):
                continue
            
            # business-related
            reg_business = re.compile(r'(?:accountant|trader|investor|banker|analyst|ceo|executive officer|entrepreneur|financial advisor|marketer)\b')
            res = reg_business.findall(substring)
            if res:
                results.append(('occupation_business',','.join(res)))

            # influencer
            reg_influencer = re.compile(r'(?:streamer|youtuber|podcaster|influencer|(?:twitch|discord) partner)')
            res = reg_influencer.findall(substring)
            if res:
                results.append(('occupation_influencer',','.join(res)))

            # healthcare
            reg_healthcare = re.compile(r'(?:dentist|doctor|nurse|physician|pharmacist|therapist|counselor|psychiatrist|dermatologist|veterinarian)\b')
            res=reg_healthcare.findall(substring)
            if res:
                res2=[]
                for phrase in res:
                    if phrase=='doctor':
                        if re.findall(r'(doctor strange|doctor who)',text):
                            continue
                    res2.append(phrase)
                results.append(('occupation_healthcare',','.join(res2)))

            # academia
            reg_academia = re.compile(r'\b(?:(?:\w+)?scientist|teacher|researcher|research assistant|scholar|educator|instructor|lecturer|prof|professor)\b')
            res=reg_academia.findall(substring)
            if res:
                results.append(('occupation_academia',','.join(res)))

            # art-related
            reg_art = re.compile(r'\b(?:artist|animator|creator|dancer|designer|dj|filmmaker|illustrator|musician|photographer|producer|rapper|singer|songwriter)\b')
            res=reg_art.findall(substring)
            if res:
                results.append(('occupation_art',','.join(res)))

            # tech
            reg_art = re.compile(r'\b(?:engineer|architect|programmer|developer|technician)\b')
            res=reg_art.findall(substring)
            if res:
                results.append(('occupation_tech',','.join(res)))

            # news & legal
            reg_news = re.compile(r'\b(?:journalist|reporter|correspondent|attorney|lawyer|spokesperson|paralegal)\b')
            res=reg_news.findall(substring)
            if res:
                results.append(('occupation_news',','.join(res)))

            # services
            reg_services = re.compile(r'\b(?:coach|attendant|colonel|lieutenant|sergeant|police officer|trainer|(?:hair)?stylist|clerk|tutor|public servant|barber|cosmetologist|(?:\w+care|social|service) (?:worker|professional))\b')
            res=reg_services.findall(substring)
            if res:
                results.append(('occupation_services',','.join(res)))

            # writing
            reg_writing = re.compile(r'\b(?:writer|blogger|editor|author|poet|publisher|playwright)\b')
            res=reg_writing.findall(substring)
            if res:
                results.append(('occupation_writing',','.join(res)))

        if results:
            return '|'.join([cat+':'+V for cat,V in results])
        else:
            return

    def extract_political(self, text):
        text=text.lower()
        results = []

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
                        results.append(('political_anticonservative','_'.join(res_negate)+'_'+phrase))
                    else:
                        results.append(('political_conservative',phrase))
            for reg in reg_lib_list:
                res=reg.findall(substring)
                for phrase in res:
                    res_negate=reg_negate.findall(substring)
                    if res_negate:
                        results.append(('political_antiliberal','_'.join(res_negate)+'_'+phrase))
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

    def extract_social_media(self, text):
        text=text.lower()
        results = []
        
        re_list = [
            re.compile(
                r'\b(insta(?:gram)?|ig|youtube|yt|cash\s?app|tumblr|twitch|venmo|web\s?(?:site|page)|whatsapp|e-?mail|snapchat|' + \
                r'onlyfans|fb|facebook|discord|tiktok|parler|mastodon)\b'),
        ]
        for reg in re_list:
            res = reg.findall(text)
            if res:
                results.extend(res)

        if results:
            results = list(set(results))
            return 'socialmedia:'+'|'.join([x.strip() for x in results])
        else:
            return

    def extract_sensitive_personal(self, text):
        text=text.lower()
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
                if flag==False:
                    results.extend(res)

        if results:
            results = list(set(results))
            return 'sensitive:'+'|'.join([x.strip() for x in results])
        else:
            return
        

def test_individual_extraction(text,identity='gender'):
    IdEx = IdentityExtactor()
    return IdEx.identity2extractor[identity](text)

def extract_identities_from_file(
    input_file:str,
    output_file:str,
    identities:list
    ):
    """Reads a file, extracts all identities, and saves it to another file
    Args:
        input_file (str): directory of input file
        output_file (str): directory of output file
        identities (list): list of identities (str) to include
    """
    print(f"""
    input: {input_file}
    output: {output_file}
    identity: {identities[0]}
    """)
    
    # create object
    IdEx = IdentityExtactor()
    
    # run each line
    if input_file.endswith('.gz'):
        f=gzip.open(input_file,'rt')
    else:
        f=open(input_file,'r')
    outf = open(output_file,'w')

    cnt = 0
    for ln,line in enumerate(f):
        line=line.split('\t')
        uid,ts,description = line[:3]

        obj = {}
        # run identity extraction
        for identity in identities:
            extractor = IdEx.identity2extractor[identity]
            res = extractor(description.lower())
            if res:
                obj[identity] = res
            
        # save results
        line_out = []
        for identity,v in obj.items():
            line_out.append(v)
        line = f'{uid}\t{ts}\t%s\n'%'|'.join(line_out)
        outf.write(line)
        if len(obj):
            cnt+=1

    f.close()
     
    print(f'Saved to {output_file} {cnt}/{ln}!')    
    # write_data_file_info(__file__,extract_identities_from_file.__name__,output_file,[input_file])
    return

def run_multiprocessing(input_dir, output_dir, modulo:int=None):
    from multiprocessing import Pool
    
    inputs = []
    all_files=sorted(os.listdir(input_dir))
    for file in all_files:
        input_file = join(input_dir,file)
        for identity in ALL_IDENTITIES:
            output_file = join(output_dir,file+'_'+identity)
            inputs.append((input_file,output_file,[identity]))

    if modulo:
        inputs = [x for i,x in enumerate(inputs) if i%7==modulo]

    pool = Pool(32)
    pool.starmap(extract_identities_from_file, inputs)
    return


            








if __name__=='__main__':
    # test 
    # text='23yr | he/him | cancer'
    # res=test_individual_extraction(text,'gender')
    # print(res)

    # input_file = '/shared/3/projects/bio-change/data/interim/description_changes/splitted/0_changes_aa'
    # output_file = '/shared/3/projects/bio-change/data/interim/description_changes/test.json.gz'
    # extract_identities_from_file(
    #     input_file,
    #     output_file,
    #     identities=['religion'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/scratch/drom_root/drom0/minje/bio-change/04.extract-identities/splitted-data')
    parser.add_argument('--output_dir', default='/scratch/drom_root/drom0/minje/bio-change/04.extract-identities/splitted-results')
    parser.add_argument('--modulo', type=int, default=None)
    args = parser.parse_args()

    run_multiprocessing(args.input_dir, args.output_dir, args.modulo)