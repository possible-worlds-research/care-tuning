from wikinlp.downloader import Downloader
from wikinlp.categories import CatProcessor
    
lang = 'simple'

def download_dump():
    # Download a simple Wikipedia corpus and process it with default options
    wikinlp = Downloader(lang)
    wikinlp.mk_wiki_data(1, tokenize=False, lower=False, doctags=True)


def download_categories(categories):
    catprocessor = CatProcessor(lang)
    catprocessor.get_category_pages(categories)
    catprocessor.get_page_content(categories)


def download_sections(categories, sections):
    catprocessor = CatProcessor(lang)
    catprocessor.get_category_pages(categories)
    catprocessor.get_page_content(categories, sections=sections, sleep_between_cats=2)


def get_all_categories():
    catprocessor = CatProcessor(lang)
    catprocessor.get_categories()

download_dump()
