"""
This script has to:

1. Separate data by langauge period
    Middle Korean - (1443-1592)
    Early Modern Korean - roughly (1593-1850)
    Contemporary Korean - (1851-2026)
2. Isolate only words that are fully in Hangul
3. Assign harmony classifications to each vowel character
    [+RTR] - 'ㅏ, ㅗ, ㆍ'
    [-RTR]- - 'ㅡ, ㅜ, ㅓ'
    NEUTRAL - 'ㅣ'
4. Isolate only vowels in each word
    e.g. '사람' -> 'ㅏㅏ'
5. Project vowels on to a tier
    This tier will lose segments as neutral vowels are excluded
6. Determine if all vowels in each word correspond to the same RTR class
    n = number of rule predictions (bigram sequences over projected tier)
    c = number of accurate predictions (bigrams that belong to the same class)
    e = exceptions (n - c)
7. Determine if n-c ratio satisfies the Tolerance Principle
    (a) Calculate tolerance threshold
        n / ln(n)
    (b) Calculate if e <= n / ln(n)
        return True or False
8. Remove one segment at a time from the vowel tier
    tier = vowels - excluded
"""