"""
The code for extracting identities from a given set of tweets
"""
import argparse
import os
from os.path import join
import re
from multiprocessing import Pool
from collections import Counter
import sys
import gzip

# import emoji
import numpy as np
import ujson as json
import pandas as pd
from tqdm import tqdm

# from twitter_identity.utils.utils import get_weekly_bins,write_data_file_info

ALL_IDENTITIES = [
    'gender', 'sexuality', 'age', 
    # 'ethnicity', 
    'religion',
    'relationship', 'education', 'occupation', 'political',
    'socialmedia', 'sensitive']

def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)

class IdentityExtractor:
    def __init__(self, include_emojis=False):
        self.include_emojis=include_emojis
        self.identity2extractor = {
            'gender':self.extract_gender,
            'sexuality':self.extract_sexuality,
            'age':self.extract_age,
            # 'ethnicity':self.extract_ethnicity,
            'religion':self.extract_religion,
            'relationship':self.extract_relationship,
            'education':self.extract_education,
            'occupation':self.extract_occupation,
            'political':self.extract_political,
            'socialmedia':self.extract_social_media,
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
        """Function for extracting phrases indicative of displayed gender

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
            results.append(('men',phrase))

        # women pronouns
        reg = re.compile(r'\b(?:she|her)\s?(?:/|\s)?\s?(?:she|hers?)\b')
        res = reg.findall(text)
        for phrase in res:
            results.append(('women',phrase))

        # nonbinary pronouns
        reg = re.compile(r'\b(?:(?:he|him|his|she|hers?|they|them|their)\s?(?:/|\s)?\s?(?:they|them|theirs?))\b')
        res = reg.findall(text)
        for phrase in res:
            results.append(('nonbinary',phrase))

        if results:
            cat2phrases={}
            for cat,phrase in results:
                if cat not in cat2phrases:
                    cat2phrases[cat]=[]
                cat2phrases[cat].append(phrase)
            for cat,V in cat2phrases.items():
                cat2phrases[cat]=','.join(list(set(V)))
            return '|'.join(['gender_'+cat+':'+V for cat,V in cat2phrases.items()])
        else:
            return

    def extract_sexuality(self, text):
        """Function for extracting phrases indicative of sexuality

        Args:
            text (_type_): Twitter bio string

        Returns:
            _type_: Formatted string listing all identities
        """
        
        results = []
        text=text.lower()

        ## step 1: for substring-level
        reg=re.compile(r'\b((?:a|pan|bi)(?:-|\s)?sexual|^(?:pan|bi|trans)$|lgbt\w+|gay|lesbian|queer|trans(?:exual|gender))\b')
        reg_negate = re.compile(r'support|advocate|friendly|sex|nsfw|content|rights?$')
        substrings = self.split_description_to_substrings(text)

        for substring in substrings:
            res = reg.findall(substring)
            if res:
                if reg_negate.findall(substring):
                    continue
                else:
                    results.extend(res)
                
        if results:
            results = list(set(results))
            return 'sexuality_lgbt:%s'%(','.join(results))
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
        
        # remove 40,000-like strings
        text = re.sub(r'[0-9]{1,3}(,[0-9]{1,3}){1,}', '', text)

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
                r'\b(african|asian|hispanic|latin)'
                # r'\b(african?|american?|asian|british|canada|canadian|mexican|england|english|european|french|indian|irish|japanese|spanish|uk|usa)\b'
            ),
            # re.compile(
            #     r'\b(🇺🇸|🇬🇧|🇨🇦|🇮🇪|🇧🇷|🇲🇽|󠁧󠁢󠁧🇯🇵|🇪🇸|🇮🇹|🇫🇷|🇩🇪|🇳🇱|🇮🇳|🇮🇩|🇹🇷|🇸🇦|🇹🇭|🇵🇭|🇦🇷|🇰🇷|🇪🇬|🇲🇾|🇨🇴)',
            # )
        ]
        reg = re_list[0]
        
        re_exclude_list = [
            re.compile(r'(hate|against|support|learn|stud(?:y|ie)|language|lingual|speak|food|dish|cuisine|culture|music|drama|tv|movie)'),
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
            return '|'.join([f'ethnicity_{x}:'+x.strip() for x in results])
        else:
            return

    def extract_religion(self, text):
        text=text.lower()
        results = []

        # re_cat = re.compile(r"\b(jesus|bible|catholic|christ(?:ian(?:ity)?)?|church|psalms?|philippians|romans|proverbs)\b")
        re_cat = re.compile(r"\b(jesus|catholic|christ(?:ian)?|church|psalm|philippians|romans|proverbs)\b")
        # re_mus = re.compile(r"\b(allah|muslim|islam(?:ic)?|quran|koran|hadith|prophet/s?muhammad)\b")
        re_mus = re.compile(r"\b(allah|muslim)\b")
        re_ath = re.compile(r"\b(atheist)\b")
        re_hin = re.compile(r"\b(hindu(?:ism)?)\b")
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
            ['cathchrist','islam','hinduism','atheism','general']):
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
                out.append(f'religion_{subcategory}:%s'%','.join(V))
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
                ['parent','partner','sibling'],
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
                out.append(f'relationship_{subcategory}:%s'%','.join(V))
            return '|'.join(out)
        else:
            return

    def extract_education(self, text):
        text=text.lower()
        results = []

        reg = re.compile(
                r'\b(ph(?:\.)?d|mba|student|univ(?:ersity)|college|grad(?:uate)?|(?:freshman|sophomore|junior|senior) at|class of|school|study(?:ing)?|\
                studie(?:s|d)|alum(?:$|na|nus|ni))\b'
            )

        re_exclude_list = [
            re.compile(r'\b(teach(?:er|ing)?|lecturer|post(?:-|\s)?doc|coach(?:ing)?|prof(?:essor)?|principal|aspiring|future)'),
            re.compile(r'student of life')
        ]

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
            return 'education_student:%s'%','.join([x.strip() for x in results])
        else:
            return

    def extract_occupation(self, text):
        """
        Follows the taxonomy from 2018 Standard Occupational Classification (SOC) taxonomy
        """
        
        text=text.lower()
        results = []
                
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            # remove false phrases
            if re.findall(r'(future|\baspir|\bex(-|\s)|former|past\b|fake)',substring):
                continue
            
            # 11-0000 Management Occupations
            reg_management = re.compile(r'\b(?:manager|director|managing (?:editor|director|partner)|ceo|(?:co(?:\s|-)?founder|entrepreneur|real estate))\b')

            # 13-0000 Business and Financial Operations Occupations
            reg_business = re.compile(r'\b(?:analyst|accountant|investor|investment)\b')
            
            # 15-0000 Computer and Mathematical Occupations
            reg_computer = re.compile(r'\b(?:dev|developer|programmer|software engineer|(?:data|cloud) architect)\b')

            # 17-0000 Architecture and Engineering Occupations
            reg_engineering = re.compile(r'\b(?:engineer(?:ing)?|architect)\b')

            # 19-0000 Life, Physical, and Social Science Occupations
            reg_science = re.compile(r'\b(?:researcher|scholar|scientist)\b')

            # 21-0000 Community and Social Service Occupations
            reg_community = re.compile(r'\b(?:social worker|counsel(?:or|ing))\b')

            # 23-0000 Legal Occupations
            reg_legal = re.compile(r'\b(?:lawyer|attorney)\b')

            # 25-0000 Educational Instruction and Library Occupations
            reg_education = re.compile(r'\b(?:educator|instructor|lecturer|prof|professor|teacher)\b')

            # 27-0000 Arts, Design, Entertainment, Sports, and Media Occupations
            reg_art = re.compile(r'\b(?:producer|actor|actress|designer|artist|singer|songwriter|coach|illustrator|musician|photographer|photography|animator|video editor|athlete|dancer|reporter|journalist|journalism|reporter|writer|author|blogger|film\s?maker|dj|trainer|influencer|twitch affiliate|streamer|podcaster|content creator)\b')

            # 29-0000 Healthcare Practitioners and Technical Occupations
            reg_healthcare = re.compile(r'\b(?:nurse|nursing|therapist|practitioner|doctor|health\s?care)\b')

            # 43-0000 Office and Administrative Support Occupations
            reg_administrative = re.compile(r'\b(?:(?:public|civil) servant)\b')

            # Get phrases            
            res=reg_management.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('management',res))
                
            res=reg_business.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('business',res))

            res=reg_computer.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('computer',res))
            
            res=reg_engineering.findall(substring)
            if res:
                if reg_computer.findall(substring):
                    pass
                elif re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('engineering',res))
            
            res=reg_science.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('science',res))
                
            res=reg_community.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('community',res))

            res=reg_legal.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b)|ace attorney)',substring):
                    pass
                else:
                    results.append(('legal',res))

            res=reg_education.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('education',res))
            
            res=reg_art.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|fav\w?|fan|fake))',substring):
                    pass
                else:
                    results.append(('art',res))

            res=reg_healthcare.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b)|doctor who)',substring):
                    pass
                else:
                    results.append(('healthcare',res))

            res=reg_administrative.findall(substring)
            if res:
                if re.findall(r'(\b(future|aspir\w+|ex(?:-|\s)|former|prev\w?|my\b))',substring):
                    pass
                else:
                    results.append(('administrative',res))


            ## Previous list
            # # academia
            # reg_academia = re.compile(r'\b(?:(?:\w+)?scientist|^i (?:teach|study)|^teaching|^research|teacher|researcher|research assistant|scholar|educator|instructor|lecturer|prof|professor)\b')
            # res=reg_academia.findall(substring)
            # if res:
            #     results.append(('academia',','.join(res)))
            
            # # business-related
            # reg_business = re.compile(r'(?:business owner|ceo|co(?:\s|-)?founder|entrepreneur|investor|)\b')
            # res = reg_business.findall(substring)
            # if res:
            #     results.append(('business',','.join(res)))
            # # reg_business = re.compile(r'(?:accountant|trader|investor|banker|analyst|ceo|executive officer|entrepreneur|financial advisor|marketer)\b')
            # # res = reg_business.findall(substring)
            # # if res:
            # #     results.append(('business',','.join(res)))

            # # influencer
            # reg_influencer = re.compile(r'(?:streamer|youtuber|podcaster|influencer|(?:twitch|discord) partner)')
            # res = reg_influencer.findall(substring)
            # if res:
            #     results.append(('influencer',','.join(res)))

            # # healthcare
            # reg_healthcare = re.compile(r'(?:dentist|doctor|nurse|physician|pharmacist|therapist|counselor|psychiatrist|dermatologist|veterinarian)\b')
            # res=reg_healthcare.findall(substring)
            # if res:
            #     res2=[]
            #     for phrase in res:
            #         if phrase=='doctor':
            #             if re.findall(r'(\bmy |doctor strange|doctor who)',text):
            #                 continue
            #         res2.append(phrase)
            #     results.append(('healthcare',','.join(res2)))

            # # art-related
            # reg_art = re.compile(r'\b(?:artist|animator|creator|dancer|designer|dj|filmmaker|illustrator|musician|photographer|producer|rapper|singer|songwriter)\b')
            # res=reg_art.findall(substring)
            # if res:
            #     results.append(('art',','.join(res)))

            # # tech
            # reg_art = re.compile(r'\b(?:engineer|architect|programmer|developer|technician)\b')
            # res=reg_art.findall(substring)
            # if res:
            #     results.append(('tech',','.join(res)))

            # # news & legal
            # reg_news = re.compile(r'\b(?:journalist|reporter|correspondent|attorney|lawyer|spokesperson|paralegal)\b')
            # res=reg_news.findall(substring)
            # if res:
            #     results.append(('news',','.join(res)))

            # # services
            # reg_services = re.compile(r'\b(?:coach|attendant|colonel|lieutenant|sergeant|police officer|trainer|(?:hair)?stylist|clerk|tutor|public servant|barber|cosmetologist|(?:\w+care|social|service) (?:worker|professional))\b')
            # res=reg_services.findall(substring)
            # if res:
            #     results.append(('services',','.join(res)))

            # # writing
            # reg_writing = re.compile(r'\b(?:writer|blogger|editor|author|poet|publisher|playwright)\b')
            # res=reg_writing.findall(substring)
            # if res:
            #     results.append(('writing',','.join(res)))
            
        out = {}
        for cat,res in results:
            if cat not in out:
                out[cat]=[]
            out[cat].extend(res)
        for cat,res in out.items():
            out[cat]=list(set(res))

        if len(out):
            return '|'.join(['occupation_'+cat+':'+','.join(V) for cat,V in out.items()])
        else:
            return

    def extract_political(self, text):
        text=text.lower()
        results = []

        # conservative pronouns
        reg_negate = re.compile(r'(?:\bnot?\b|\bnever|hat(?:e|red|ing)|\banti|\bex\b|idiot|stupid|dumb|fool|wrong|enemy|worst|dump|dislike|detest|despise|troll|impeach|imprison|fuck|danger|threat|terrible|horrible|survive|shit)')
        reg_con_list=[
            # re.compile(r'\b(?:maga(?:2020)?|(?:1|2)a|kag(?:2020)?|build\s?the\s?wall|america\s?first|\bpro(?:\s|-)?life|make\s?america\s?great\s?again)\b'),
            re.compile(r'\b(?:conservative|maga|libertarian|right(?:-|\s)?(?:ist|wing)|republican)\b'),
            # re.compile(r'\b(?:(?:neo-?)?conservative|traditionalist|nationalist|libertarian|right(?:-|\s)?(?:ist|wing|republican))\b'),
        ]

        reg_lib_list=[
            re.compile(
                r'\b(?:#bidenharris(?:\w+)?|liberal|progressive|democrat|left(?:\s|-)?(?:wing|ist))\b'),
            # re.compile(
            #     r'\b(?:#bidenharris(?:\w+)?|liberal|progressive|socialist|democrat|left(?:\s|-)?(?:wing|ist)|blue\s?wave|equal\s?(?:ist|ism|ity|right)|marxist)\b'),
        ]
        
        substrings = self.split_description_to_substrings(text)
        for substring in substrings:
            for reg in reg_con_list:
                res=reg.findall(substring)
                for phrase in res:
                    res_negate=reg_negate.findall(substring)
                    if res_negate:
                        results.append(('anticonservative','_'.join(res_negate)+'_'+phrase))
                    else:
                        results.append(('conservative',phrase))
            for reg in reg_lib_list:
                res=reg.findall(substring)
                for phrase in res:
                    res_negate=reg_negate.findall(substring)
                    if res_negate:
                        results.append(('antiliberal','_'.join(res_negate)+'_'+phrase))
                    else:
                        results.append(('liberal',phrase))

            reg_activism_list=[
                re.compile(r'(?:black\s?lives\s?matter|blm)'),
                re.compile(f'\b(?:acab|activism|activist|feminist|feminism|#resist)\b')
            ]
            for reg in reg_activism_list:
                res=reg.findall(substring)
                for phrase in res:
                    results.append(('activism',phrase))

        if results:
            cat2phrases={}
            for cat,phrase in results:
                if cat not in cat2phrases:
                    cat2phrases[cat]=[]
                cat2phrases[cat].append(phrase)
            for cat,V in cat2phrases.items():
                cat2phrases[cat]=','.join(list(set(V)))
            return '|'.join(['political_'+cat+':'+V for cat,V in cat2phrases.items()])
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
            return 'personal_socialmedia:'+','.join([x.strip() for x in results])
        else:
            return

    def extract_sensitive_personal(self, text):
        text=text.lower()
        results = []
        
        re_list = [
            re.compile(r'\b(?:depress(?:ed|ion)|autis(?:m|tic)|anxiety|adhd)\b')
                # r'\b(surviv(?:ed|or)|depress(?:ed|ion)|autis(?:m|tic)|anxiety|adhd|diabetes|fibromyalgia|cancer|trauma(?:tized|tizing)?|' + \
                # r'victim|brain injury|strokes|ptsd|chronic (?:pain|illness)|jobless|homeless|unemployed|disorder|dyslexi(?:a|c))\b')
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
            return 'personal_sensitive:'+','.join([x.strip() for x in results])
        else:
            return
        


def test_individual_extraction(text,identity='gender'):
    IdEx = IdentityExtractor()
    return IdEx.identity2extractor[identity](text)

# def extract_identities_from_file(
#     input_file:str,
#     output_file:str,
#     identities:list
#     ):
#     """Reads a file, extracts all identities, and saves it to another file
#     Args:
#         input_file (str): directory of input file
#         output_file (str): directory of output file
#         identities (list): list of identities (str) to include
#     """
#     print(f"""
#     input: {input_file}
#     output: {output_file}
#     identity: {identities[0]}
#     """)
    
#     # create object
#     IdEx = IdentityExtactor()
    
#     # run each line
#     if input_file.endswith('.gz'):
#         f=gzip.open(input_file,'rt')
#     else:
#         f=open(input_file,'r')
#     outf = gzip.open(output_file,'wt')

#     cnt = 0
#     ts = input_file.split('.')[-3]
#     for ln,line in enumerate(f):
#         obj0 = json.loads(line)
#         uid,description = obj0['id_str'],obj0['description']

#         obj = {}
#         # run identity extraction
#         if type(description)==str:
#             for identity in identities:
#                 extractor = IdEx.identity2extractor[identity]
#                 res = extractor(description)
#                 if res:
#                     obj[identity] = res.lower()
            
#         # save results
#         line_out = []
#         for identity,v in obj.items():
#             line_out.append(v)
#         line = f'{uid}\t{ts}\t%s\n'%'|'.join(line_out)
#         outf.write(line)
#         if len(obj):
#             cnt+=1

#     f.close()
     
#     print(f'Saved to {output_file} {cnt}/{ln}!')    
#     # write_data_file_info(__file__,extract_identities_from_file.__name__,output_file,[input_file])
#     return

def extract_identities_from_file(input_file,output_file):
    # load extractor
    IdEx = IdentityExtractor()
    
    # get number of lines
    with gzip.open(input_file,'rt') as f:
        for ln,_ in enumerate(f):
            continue
    n_lines = ln
    
    # get phrases
    cnt=0
    with gzip.open(input_file,'rt') as f,\
        gzip.open(output_file,'wt') as outf:
            for line in tqdm(f,total=n_lines):
                cnt+=1
                # if cnt>=10000:
                #     break
                uid,dt,desc=line.split('\t')
                desc=desc.lower().strip()
                obj = {}
                for identity in ALL_IDENTITIES:
                    extractor = IdEx.identity2extractor[identity]
                    res = extractor(desc)
                    if res:
                        obj[identity]=res.lower()
                res = sorted(list(obj.values()))
                outf.write('\t'.join([uid,dt,desc]+res)+'\n')                    
    return
    

def run_multiprocessing(input_dir, output_dir, identities=ALL_IDENTITIES, modulo:int=None):
    from multiprocessing import Pool
    
    inputs = []
    all_files=sorted(os.listdir(input_dir))
    for file in all_files:
        input_file = join(input_dir,file)
        for identity in identities:
            output_file = join(output_dir,file+'_'+identity)
            inputs.append((input_file,output_file,[identity]))

    if modulo:
        inputs = [x for i,x in enumerate(inputs) if i%10==modulo]

    pool = Pool(32)
    # pool.starmap(extract_identities_from_file, inputs)
    for input in inputs:
        extract_identities_from_file(*input)
    return

def set_multiprocessing(fun, load_dir, save_dir, modulo=None):
    pool=Pool(32)
    from os import sched_getaffinity
    n_available_cores = len(sched_getaffinity(0))
    print(f'Number of Available CPU cores: {n_available_cores}')
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(f'Number of CPU cores: {number_of_cores}')


    files = sorted([file for file in os.listdir(load_dir) if file.startswith('user')])

    if type(modulo)==str:
        modulo=int(modulo)
        files=[files[i] for i in range(len(files)) if i%10==modulo]
    print(len(files),' files to read!')

    inputs = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for twitter_file in files:
        inputs.append((join(load_dir,twitter_file), join(save_dir,twitter_file),ALL_IDENTITIES))
    # try:
    # pool.map(collect_tweets,files)
    pool.starmap(fun,inputs)
    # fun(*inputs[10])
    # finally:
    pool.close()
    return

def post_processing(input_file,output_file):
    out={}
    unique_ids={}
    # get all identities of a user
    with gzip.open(input_file,'rt') as f:
        for line in f:
            line=line.split('\t')
            uid,dt,desc=line[:3]
            if uid not in out:
                out[uid]=[]
                unique_ids[uid]=[]
            out[uid].append(line[1:])
            unique_ids[uid].extend([v.split(':')[0] for v in line[3:]])
    
    # get only users with no conflicting identities
    with gzip.open(output_file,'wt') as outf:
        for uid,V in unique_ids.items():
            S = set(V) # unique ids through all profiles
            # political
            if ('political_liberal' in S) and ('political_conservative' in S):
                continue
            if ('political_antiliberal' in S) or ('political_anticonservative' in S):
                continue
            # age count
            age_cnt=0
            for age in ['13-17','18-24','25-34','35-49','50+']:
                if f'age_{age}' in S:
                    age_cnt+=1
            if age_cnt>1:
                continue
            # gender pronouns
            gender_cnt=0
            for gender in ['men','women','nonbinary']:
                if f'gender_{gender}' in S:
                    gender_cnt+=1
            if gender_cnt>1:
                continue
            
            lines = out[uid]
            for line in lines:
                outf.write('\t'.join([uid]+line))
    
    return

if __name__=='__main__':
    
    # extract identities from the files
    # input_file='/shared/3/projects/bio-change/data/interim/description_changes/03.filtered'
    # output_file='/shared/3/projects/bio-change/data/interim/description_changes/04.identity-extracted'
    # extract_identities_from_file(input_file,output_file)    
    
    # post-processing: removing cases where 2+ conflicting identities appear in the same person
    input_file='/shared/3/projects/bio-change/data/interim/description_changes/04.identity-extracted/description_changes_0_changes.tsv.gz'
    output_file='/shared/3/projects/bio-change/data/interim/description_changes/05.post-processed/description_changes_0_changes.tsv.gz'
    post_processing(input_file,output_file)    
    input_file='/shared/3/projects/bio-change/data/interim/description_changes/04.identity-extracted/description_changes_1_change.tsv.gz'
    output_file='/shared/3/projects/bio-change/data/interim/description_changes/05.post-processed/description_changes_1_change.tsv.gz'
    post_processing(input_file,output_file)