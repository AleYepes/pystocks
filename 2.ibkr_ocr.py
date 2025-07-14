import pandas as pd
from time import sleep
from datetime import datetime, timedelta
import traceback
import random
from tqdm.auto import tqdm
import pyautogui
import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz import fuzz
import gc
import re
from functions import *

import platform
if platform.system() == "Darwin":
    from AppKit import NSWorkspace, NSApplicationActivateIgnoringOtherApps


# Set-up functions
def extract_text(ocr, screenshot):
    try:
        result = ocr.ocr(screenshot, cls=False)
    except Exception as e:
        print(f"Error during OCR: {e}")
        result = [None]

    text = []
    if result[0]:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                if line[-1][-1] > 0.85:
                    text.append(line[-1][0].replace(' ',''))
    # print(result)
    return text

def scroll(amount, attempts=2):
    SCROLL_TOP = 600
    SCROLL_BOT = 550
    
    before = pyautogui.screenshot(region=(0, 25, 1919, 1054))
    before = np.array(before)
    for _ in range(attempts):
        rand_int = random.randint(SCROLL_BOT, SCROLL_TOP)
        pyautogui.scroll(amount, rand_int, rand_int)
        after = pyautogui.screenshot(region=(0, 25, 1919, 1054))
        after = np.array(after)

        if not np.array_equal(before, after):
            return
        
    # raise Exception(f'scroll() failed')
        
def wait(seconds=5, interval=0.5):
    start = datetime.now()
    timeout = timedelta(seconds=seconds)
    sleep(interval)
    while datetime.now() - start < timeout:
        screenshot = pyautogui.screenshot(region=(10, 280, 440, 200))
        screenshot = np.array(screenshot)
        screenshot = screenshot[:, :, :3]

        if np.any(np.all(screenshot == (247, 247, 247), axis=-1)):
            return
        sleep(interval)
    raise Exception('wait() text never loaded')
    
def select_tab(name, seconds=5):
    try:
        scroll(999)
        tab = pyautogui.locateCenterOnScreen(f'assets/{name}/tab_label.png', confidence=0.9)
        pyautogui.click(tab)
        wait(seconds)
        return True
    except Exception:
        return False
    
def agree():
    while True:
        try:
            agree = pyautogui.locateOnScreen("assets/fund_setup/agree.png", confidence=0.9)
            pyautogui.click(agree)
        except Exception:
            break

def clear_modals():
    while True:
        try:
            log_off_timer = pyautogui.locateOnScreen("assets/fund_setup/log_off_timer.png", confidence=0.9)
            left_x = log_off_timer.left + 34
            center_y = log_off_timer.top + log_off_timer.height // 2
            pyautogui.click(left_x, center_y)
        except Exception:
            break
    
    while True:
        try:
            search_error = pyautogui.locateOnScreen("assets/fund_setup/search_error.png", confidence=0.9)
            left_x = search_error.left + 25
            center_y = search_error.top + search_error.height // 2
            pyautogui.click(left_x, center_y)
        except Exception:
            break

def select_exchange():
    while True:
        try:
            pyautogui.locateOnScreen("assets/fund_setup/contract_selection.png", confidence=0.9)
            pyautogui.press("up")
            pyautogui.press("enter")
            return True
        except Exception:
            break

def switch_to_app(app_name='Trader Workstation'):
    workspace = NSWorkspace.sharedWorkspace()
    apps = workspace.runningApplications()
    for app in apps:
        if app.localizedName() == app_name:
            app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            break

def exit_search():
    pyautogui.press("enter", presses=2, interval=0.3)
    if select_exchange():
        exchange_bug = True
    else:
        exchange_bug = False
    agree()

    # Redundant check in case a press fails
    pyautogui.press("enter", presses=2, interval=0.3)
    if not exchange_bug and select_exchange():
        exchange_bug = True
    else:
        exchange_bug = False
    agree()
    clear_modals()
    return exchange_bug

# Search functions
def check_search_results(ocr, row, screenshot, screenshot_left, screenshot_top, buffer, width=800, target_color=[64, 64, 64, 255], background_color=[109, 111, 113, 255]):
    matches, HEIGHT, adjustable_height, max_adjustable_height, top, row_text_detected_previously = [], 21, 21, 42, buffer, False

    screenshot_array = np.array(screenshot)
    mask = np.all(screenshot_array == target_color, axis=-1)
    screenshot_array[mask] = background_color
    
    while True:
        # display(Image.fromarray(screenshot_array[top:top+adjustable_height]))
        text_list = extract_text(ocr, screenshot_array[top:top+adjustable_height])
        # print(text_list, adjustable_height)
        # print(row['symbol'], row['validExchanges'])
        
        if not text_list:
            if row_text_detected_previously:
                pass
            else:
                break   

        if len(text_list) > 1:
            search_symbol = text_list[0]
            row_symbol = str(row['symbol'])
            try:
                int(row_symbol)
                confidence = 90
            except Exception:
                confidence = 75

            if (len(search_symbol) == len(row_symbol)) and fuzz.partial_ratio(search_symbol, row_symbol) >= confidence:
                search_exchange = text_list[1:]
                search_exchange = [exchange for exchange in search_exchange if '(' in exchange]
                position = (screenshot_left + (width / 2), (screenshot_top + top) + adjustable_height / 2)
                matches.append((search_exchange, search_symbol, position))
            # Move to the next row
            top += HEIGHT + (adjustable_height - HEIGHT)//2
            adjustable_height = HEIGHT
            row_text_detected_previously = False
        else:
            # Adjust region for better OCR
            top -= 1
            adjustable_height += 2
            row_text_detected_previously = True
            if adjustable_height > max_adjustable_height:
                top += HEIGHT + (adjustable_height - HEIGHT)//2
                adjustable_height = HEIGHT
                row_text_detected_previously = False

    if matches: 
        for match in matches: # Solely check exchange matches first
            for ex in match[0]:
                if fuzz.partial_ratio(row['exchange'], ex) >= 80:
                    return ex, match[1], match[2]
                
        for match in matches: # Then consider primaryExchange matches
            for ex in match[0]:
                if fuzz.partial_ratio(row['primaryExchange'], ex) >= 80:
                    return ex, match[1], match[2]
                
        valid_exchanges = row['validExchanges'].split(',') if row['validExchanges'] else []
        for match in matches: # Finally consider validExchange matches
            for ex in match[0]:
                for valid_exchange in valid_exchanges:
                    if fuzz.partial_ratio(valid_exchange.strip(), ex) >= 80:
                        return ex, match[1], match[2]
                    
        # match = matches[0] # If no match, return first ocr result
        # return np.nan, np.nan, match[2]
    return None, None, None
        
def check_search_table(buffer, width=800, seconds=8, target_color=[64, 64, 64, 255], background_color=[109, 111, 113, 255]):
    start = datetime.now()
    timeout = timedelta(seconds=seconds)

    i = 0.0
    while datetime.now() - start < timeout:
        try:
            table_heading = pyautogui.locateOnScreen(f'assets/fund_setup/search{round(i)}.png', confidence=0.8)
            break
        except Exception:
            i += 1/3

    left = table_heading.left
    top = table_heading.top + table_heading.height - buffer # add some buffer for OCR region adjustments in check_search_results()
    try:
        table_bottom = pyautogui.locateOnScreen(f'assets/fund_setup/search_bottom.png', confidence=0.9)
        table_height = (table_bottom.top - top) + table_bottom.height + (buffer - 21)
    except Exception:
        print('set height')
        table_height = 300

    next_row = buffer + 1 # +1 to make sure that we're detecting the next row
    screenshot = pyautogui.screenshot(region=(left, top, width, table_height))
    screenshot_array = np.array(screenshot)
    if screenshot_array[next_row][-1].tolist() == target_color:
        return screenshot, left, top, buffer
    elif screenshot_array[next_row][-1].tolist() == background_color:
        raise Exception('No search results found')
    else:
        return check_search_table(buffer, seconds=seconds, target_color=target_color, background_color=background_color)
        
def search_eft(ocr, row, wait_time=5):
    scroll(999)
    pyautogui.press("esc")
    pyautogui.click((1880,100), interval=0.2)

    pyautogui.click(POSITIONS['search_box'], interval=0.2)
    pyautogui.write(row['longName'])
    pyautogui.press("enter")

    buffer = 15
    target_color = [64, 64, 64, 255]
    background_color = [109, 111, 113, 255]
    screenshot, left, top, buffer = check_search_table(buffer, seconds=wait_time, target_color=target_color, background_color=background_color)
    
    exchange, symbol, row_position = check_search_results(ocr, row, screenshot, left, top, buffer + 1, target_color=target_color, background_color=background_color) # +1 to center the row text in frame. 
    if row_position:
        pyautogui.click(row_position)
        exchange_bug = exit_search()
        return exchange, symbol, exchange_bug
    else:
        clear_modals()
        raise Exception('Failed to detect the symbol in the search results')
    
def quick_search_etf(row, count=None, name=None):
    pyautogui.press("esc")
    pyautogui.click((1880,100), interval=0.2)
    pyautogui.click(POSITIONS['search_box'], interval=0.2)
    if count:
        pyautogui.press("delete", presses=count)
        pyautogui.press("backspace", presses=count)
    if name:
        pyautogui.write(name)
    else:
        pyautogui.write(row['symbol'])

    exchange_bug = exit_search()
    return exchange_bug

# Profile functions
def process_profile(text_list):
    headings = ['TotalExpenseRatio', 'TotalNetAssets', 'BenchmarkIndex', 'Domicile', 'MarketGeoFocus', 'MarketCapFocus', 'FundCategory']
    current_label, current_values, labels, values = None, [], [], []
    threshold = 80

    for item in text_list:
        matches = [(heading, fuzz.partial_ratio(item, heading)) for heading in headings]
        best_match = max(matches, key=lambda x: x[1])

        if best_match[1] >= threshold and (current_label != best_match[0]):
            if current_label:
                labels.append(current_label)
                values.append(' '.join(current_values))
                current_values = []
            current_label = best_match[0]
        else:
            current_values.append(item)

    if current_label:
        labels.append(current_label)
        values.append(' '.join(current_values))

    return list(zip(labels, values))

def extract_profile(ocr):
    profile = pyautogui.locateOnScreen("assets/overview/profile.png", confidence=0.9)
    left = profile.left
    top = profile.top + profile.height
    lipper = pyautogui.locateOnScreen("assets/overview/lipper.png", confidence=0.9)
    width = 600
    height = lipper.top - top

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)

    text_list = extract_text(ocr, screenshot)
    scroll(-height/8)
    if text_list:
        return process_profile(text_list)
    else:
        raise Exception('skip')
    
def check_title(ocr, title, seconds=5, interval=1):
    start = datetime.now()
    timeout = timedelta(seconds=seconds)
    sleep(interval)
    while datetime.now() - start < timeout:

        # Check for white text
        screenshot = pyautogui.screenshot(region=(25, 100, 65, 25))
        screenshot = np.array(screenshot)
        text_color = (247, 247, 247, 255)
        if np.any(np.all(screenshot == text_color, axis=-1)):
            
            # Check text match
            screenshot = pyautogui.screenshot(region=(25, 100, 480, 60))
            screenshot = np.array(screenshot)
            text = extract_text(ocr, screenshot)
            if fuzz.partial_ratio(text[0][:len(title)].upper(), title.upper()) >= 85:
                return text[0]
            else:
                return
        sleep(interval)

def check_tradable(seconds=5, interval=0.5):
    timeout = timedelta(seconds=seconds)
    start = datetime.now()
    sleep(interval)
    while datetime.now() - start < timeout:

        # Check for white text
        screenshot = pyautogui.screenshot(region=(25, 100, 65, 25))
        screenshot = np.array(screenshot)
        text_color = (247, 247, 247, 255)
        if np.any(np.all(screenshot == text_color, axis=-1)):

            # Check for nt sign
            sleep(interval)
            screenshot = pyautogui.screenshot(region=(25, 125, 300, 100))
            screenshot = np.array(screenshot)
            nt_sign_color = (240, 71, 80, 255)
            if np.any(np.all(screenshot == nt_sign_color, axis=-1)):
                return False
            else:
                return True

def process_holding_types(text_list):
    # Assumes text is identified from left to right, and top to bottom
    for i, element in enumerate(text_list):
        if element.strip().isupper():
            text_list = text_list[:i]
            break

    last_label, labels , values = None, [], []
    for item in (text_list):
        if is_numerical(item):
            labels.append(last_label)
            values.append(item)
            last_label = None
        else:
            if last_label:
                labels.append(last_label)
                values.append(None)
            last_label = item

    return list(zip(labels, values))

def extract_holding_types(ocr):
    holdings = pyautogui.locateOnScreen("assets/overview/holdings.png", confidence=0.9)
    left = holdings.left
    top = holdings.top + holdings.height
    dividends = pyautogui.locateOnScreen("assets/overview/dividends.png", confidence=0.9)
    width = 600
    height = dividends.top - top

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)

    text_list = extract_text(ocr, screenshot)
    scroll(-height / 8)
    if len(text_list) > 1:
        return process_holding_types(text_list)
    else:
        raise Exception('skip')

def process_dividends(text_list):
    # Assumes text is identified from left to right, and top to bottom
    labels, values, label_indicator, value_indicator = [], [], False, False
    
    for item in text_list:
        if fuzz.partial_ratio(item, 'Div.YieldTTM') >= 90:
            label_indicator = True
        if label_indicator:
            if item[-1] == '%' or item == 'Unknown':
                value_indicator = True
                label_indicator = False
                values.append(item)
            else:
                labels.append(item)
        elif value_indicator:
            values.append(item)

    return list(zip(labels, values))
    
def extract_style():
    style = pyautogui.locateOnScreen("assets/overview/style_matrix.png", confidence=0.85)
    left = style.left + style.width + 70
    width = 280
    height = 172
    lipper = pyautogui.locateOnScreen("assets/overview/lipper.png", confidence=0.85)
    top = lipper.top - height - 113

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)
    highlight_colors = [
        (29, 51, 88, 255),
        (255, 255, 255, 255),
        (41, 112, 234, 255)
    ]
    background_color = (24, 24, 24, 255)

    color_matches = np.zeros(screenshot.shape[:-1], dtype=bool)
    for color in highlight_colors:
        color_matches |= np.all(screenshot == color, axis=-1)

    if np.any(color_matches):
        styles = []
        rows = ['large', 'multi', 'mid', 'small']
        columns = ['value', 'core', 'growth']
        for i, row in enumerate(rows):
            row_step_px = round(height / (len(rows) - 1)) - 1 # First -1 to get the num of internal boundaries == num of areas - 1. Second -1 to avoid index overflow
            for j, col in enumerate(columns):
                col_step_px = round(width / (len(columns) - 1)) - 1
                pixel = screenshot[row_step_px * i][col_step_px * j].tolist()
                styles.append((f'{row}-{col}', pixel != list(background_color)))
    
        return styles

def process_lipper(text_index, screenshot, width, height):
    bg_color = [24, 24, 24, 255]
    missing_color = [0, 0, 0, 255]
    row_idx = 16 + round(35 * text_index)
    for j in range(5):
        col_step_px = round(width/5) - 1
        pixel = screenshot[row_idx][col_step_px * (j+1)].tolist()
        if pixel == bg_color or pixel == missing_color:
            return j + 1

def extract_lipper(ocr):
    try:
        lipper = pyautogui.locateOnScreen("assets/overview/lipper.png", confidence=0.9)
        left = lipper.left
        top = lipper.top + lipper.height
        holdings = pyautogui.locateOnScreen("assets/overview/holdings.png", confidence=0.9)
        lipper_width = 300
        height = holdings.top - top

        screenshot = pyautogui.screenshot(region=(left, top, lipper_width, height))
        screenshot = np.array(screenshot)
        text_list = extract_text(ocr, screenshot)
        if text_list:
            width = 285
            screenshot = pyautogui.screenshot(region=(left+lipper_width+24, top+12, width, height-12))
            screenshot = np.array(screenshot)
            
            lipper = []
            for i, label in enumerate(text_list):
                value = process_lipper(i, screenshot, width, 34)
                lipper.append((label, value))
            scroll(-height/10)
            return lipper
        scroll(-height/10)
    except Exception as e:
        raise Exception(f'extract_dividends() {e}')
    
# Holding functions
def show_more(type=1):
    try:
        if type != 1:
            show_more = pyautogui.locateCenterOnScreen("assets/holdings/show_more2.png", confidence=0.9)
        else:
            show_more = pyautogui.locateCenterOnScreen("assets/holdings/show_more.png", confidence=0.9)
        pyautogui.click(show_more)
    except Exception:
        pass

def process_top10(text_list):
    # Assumes text is identified from left to right, and top to bottom
    index, current_labels, labels , values = True, [], [], []

    for item in text_list:
        if index and len(item) <= 2 and not item.endswith('%'):
            index = False 
        elif is_numerical(item) and item.endswith('%'):
            labels.append('-'.join(current_labels))
            values.append(item)
            current_labels = []
            index = True
        else:
            current_labels.append(item)
            index = False 
    return list(zip(labels, values))

def extract_top10(ocr):    
    top10 = pyautogui.locateOnScreen("assets/holdings/top10.png", confidence=0.9)
    left = top10.left
    top = top10.top + top10.height
    width = 626
    height = 455

    return capture_text(ocr, process_top10, left, top, width, height)

def process_industry(text_list): # Industry displays labels, value pairs backwards, so ocr reads them backwards
    last_value, labels , values = None, [], []
    for item in (text_list):
        if is_numerical(item):
            if last_value:
                labels.append(None)
                values.append(last_value)
            last_value = item
        else:
            labels.append(item)
            values.append(last_value)
            last_value = None
    return list(zip(labels, values))

def process_holding_tables(text_list):
    last_value, labels , values = None, [], []
    for item in (text_list):
        value = re.sub(r'[^\d.%]', '', item)
        if is_numerical(value):
            labels.append(last_value)
            values.append(value)
            last_value = None
        else:
            if last_value:
                labels.append(last_value)
                values.append(None)
            last_value = item
    return list(zip(labels, values))

def process_bonds(text_list):
    last_value, labels , values = None, [], []
    for item in (text_list):
        if is_numerical(item):
            labels.append(last_value)
            values.append(item)
            last_value = None
        else:
            if last_value:
                labels.append(last_value)
                values.append(None)
            last_value = item
    return list(zip(labels, values))

def extract_industry(ocr):
    show_more()
    industry = pyautogui.locateOnScreen("assets/holdings/industry.png", confidence=0.9)
    scroll(-(industry.height*3/4))

    industry = pyautogui.locateOnScreen("assets/holdings/industry.png", confidence=0.9)
    left = industry.left + 40
    top = industry.top + industry.height
    width = 550
    try:
        show_less = pyautogui.locateOnScreen("assets/holdings/show_less.png", confidence=0.9)
        height = show_less.top - top
    except Exception:
        height = 450

    return capture_text(ocr, process_industry, left, top, width, height)

def extract_country(ocr):
    country = pyautogui.locateOnScreen("assets/holdings/country.png", confidence=0.9)
    scroll(-(country.top/15))
    show_more(2)
    country = pyautogui.locateOnScreen("assets/holdings/country.png", confidence=0.9)
    left = country.left + 50
    top = country.top + country.height
    width = 460
    currency = pyautogui.locateOnScreen("assets/holdings/currency.png", confidence=0.9)
    height = currency.top - top

    return capture_text(ocr, process_holding_tables, left, top, width, height)

def extract_currency(ocr):
    currency = pyautogui.locateOnScreen("assets/holdings/currency.png", confidence=0.9)
    scroll(-(currency.top/20))
    show_more(2)
    scroll(-(currency.top/50), 1)
    currency = pyautogui.locateOnScreen("assets/holdings/currency.png", confidence=0.9)
    left = currency.left + 50
    top = currency.top + currency.height
    width = 460
    try:
        debtor = pyautogui.locateOnScreen("assets/holdings/debtor_quality.png", confidence=0.9)
        height = debtor.top - top
    except Exception:
        height = BOTTOM - top

    return capture_text(ocr, process_holding_tables, left, top, width, height)

# Resize function
def capture_text(ocr, function, left, top, width, height):
    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)
    text_list = extract_text(ocr, screenshot)
    if text_list:
        try:
            return function(text_list)
        except:
            expand_px = 4
            screenshot = screenshot = pyautogui.screenshot(region=(left - expand_px, top - expand_px, width + expand_px*2, height + expand_px*2))
            screenshot = np.array(screenshot)
            text_list = extract_text(ocr, screenshot)
            if text_list:
                return function(text_list)
            
# Bond functions
def extract_debtors(ocr, name):
    scroll(-499, 1)
    debtor = pyautogui.locateOnScreen(f"assets/holdings/{name}.png", confidence=0.9)
    left = debtor.left
    top = debtor.top + debtor.height
    width = 405
    height = BOTTOM - top

    return capture_text(ocr, process_bonds, left, top, width, height)

# Fundamentals functions
def extract_fundamentals_text(ocr, screenshot):
    # display(Image.fromarray(screenshot))
    try:
        results = ocr.ocr(screenshot, cls=False)
    except Exception as e:
        print(f"Error during OCR: {e}")
        results = [None]

    text_list = []
    if results and results[0]:
        for res in results:
            for line in res:
                bbox, (text, conf) = line[0], line[-1]
                if conf > 0.85:
                    text_list.append({
                        'text': text.replace(' ', ''),
                        'bbox': bbox,
                        'conf': conf
                    })
    return text_list

def calculate_value_crop(label_bbox, screenshot, offset=5, expansion=0):
    """
    Given a label's bounding box, calculates a region where its value should be.
    The region is expanded by 'expansion' pixels on all sides.
    """
    # Extract coordinates from label_bbox (assumed to be list of 4 points)
    x_coords = [pt[0] for pt in label_bbox]
    y_coords = [pt[1] for pt in label_bbox]
    label_left, label_top = min(x_coords), min(y_coords)
    label_right, label_bottom = max(x_coords), max(y_coords)
    
    # Initial value region: to the right of the label with a small offset.
    initial_value_left = label_right + offset
    initial_value_top = label_top
    screenshot_width = screenshot.shape[1]
    initial_value_width = screenshot_width - initial_value_left
    initial_value_height = label_bottom - label_top

    # Expand the region by 'expansion' pixels on all sides.
    new_x = max(initial_value_left, 0)
    new_y = max(initial_value_top - expansion, 0)
    new_width = initial_value_width
    new_height = initial_value_height + 2 * expansion

    # Ensure the region stays within the screenshot boundaries.
    if new_x + new_width > screenshot.shape[1]:
        new_width = screenshot.shape[1] - new_x
    if new_y + new_height > screenshot.shape[0]:
        new_height = screenshot.shape[0] - new_y

    return (int(new_x), int(new_y), int(new_width), int(new_height))

def detect_value_with_expansion(ocr, screenshot, label_bbox, initial_offset=5, max_expansion=49):
    """
    Reattempts to detect value by progressively expanding the crop region.
    Returns the first_pass detected text or None if no detection is made within max_expansion.
    """
    expansion = 0
    new_det = []
    while not new_det and expansion <= max_expansion:
        crop_region = calculate_value_crop(label_bbox, screenshot, offset=initial_offset, expansion=expansion)
        cropped = screenshot[crop_region[1]:crop_region[1]+crop_region[3],
                             crop_region[0]:crop_region[0]+crop_region[2]]
        new_det = extract_fundamentals_text(ocr, cropped)
        if new_det:
            break
        expansion += 1
    return new_det[0]['text'] if new_det else None

def process_fundamentals(detections, screenshot, ocr):
    last_label, last_bbox = None, None
    labels, values = [], []
    
    for det in detections:
        text = det['text']
        bbox = det['bbox']
        if is_numerical(text) or text.isupper():
            if last_label is not None:
                labels.append(last_label)
                values.append(text)
                last_label = None  # reset after pairing
        else:
            if text == 'Equity':
                continue
            elif last_label:
                labels.append(last_label)
                new_value = detect_value_with_expansion(ocr, screenshot, last_bbox, initial_offset=5)
                values.append(new_value)
            last_label = text
            last_bbox = bbox
    
    # Handle a leftover label.
    if last_label and last_label != 'Equity':
        labels.append(last_label)
        new_value = detect_value_with_expansion(ocr, screenshot, last_bbox, initial_offset=5)
        values.append(new_value)

    return list(zip(labels, values))

def extract_fundamentals(ocr, prev_list=None):
    first_pass = False
    try:
        top_screenshot_boundary = pyautogui.locateOnScreen("assets/fundamentals/metric.png", confidence=0.9)
        first_pass = True
    except pyautogui.ImageNotFoundException:
        top_screenshot_boundary = pyautogui.locateOnScreen("assets/fundamentals/top_border.png", confidence=0.9)
    left = top_screenshot_boundary.left
    top = top_screenshot_boundary.top + top_screenshot_boundary.height
    width = 550
    height = BOTTOM - top

    screenshot = pyautogui.screenshot(region=(left, top, width, height))
    screenshot = np.array(screenshot)

    if first_pass:
        title = pyautogui.locateOnScreen("assets/fundamentals/title.png", confidence=0.9)
        title_screenshot = pyautogui.screenshot(region=(left, title.top + title.height, title.width, title.height))
        funds_date = extract_text(ocr, np.array(title_screenshot))
    
    text_list = extract_fundamentals_text(ocr, screenshot)
    if text_list:
        if len(text_list) <= 15:
            return process_fundamentals(text_list, screenshot, ocr), ' '.join(funds_date)
        else:
            current_list = process_fundamentals(text_list, screenshot, ocr)
            if prev_list:
                if set(current_list) == set(prev_list):
                    return current_list
                return list(set(current_list + prev_list))
            scroll(-999)
            return extract_fundamentals(ocr, current_list), ' '.join(funds_date)
    raise Exception('extract_fundamentals() error')

def get_remaining():
    contract_details = load('data/contract_details.csv')
    # contract_details = contract_details[contract_details['currency'] == 'EUR']
    try:
        final_df = load('data/contract_elaborated.csv')
        final_df = final_df[final_df.apply(is_row_valid, axis=1)]
        final_df['date_scraped'] = pd.to_datetime(final_df['date_scraped'])
        final_df['days_since_last_scrape'] = (datetime.now() - final_df['date_scraped']).dt.days

        # exclusion_condition = (final_df['exchange_bug'] == True) | (final_df['exact_search'] == True) | (~final_df['profile'].isna()) | (final_df['days_since_last_scrape'] <= 30)
        exclusion_condition = (final_df['exchange_bug'] == True) | ((final_df['exact_search'] == True) & (final_df['days_since_last_scrape'] <= 30))# | (~final_df['profile'].isna())
        symbols_to_exclude = final_df[exclusion_condition]['conId']
        remaining = contract_details[~contract_details['conId'].isin(symbols_to_exclude)]

        # # To debug invalid rows
        # remaining = final_df.copy()
        # remaining = remaining[~remaining.apply(is_row_valid, axis=1)]
    except FileNotFoundError:
        remaining = contract_details.copy()
        
    remaining = remaining[~remaining['longName'].apply(has_bad_multiplier)]
    remaining = remaining[['symbol', 'exchange', 'primaryExchange', 'validExchanges', 'currency', 'conId', 'longName', 'stockType', 'isin']]
    remaining = (remaining
                 .assign(currency_is_euro=remaining['currency'] == 'EUR')
                 .sort_values(by='currency_is_euro', ascending=False)
                 .drop(columns='currency_is_euro')
                 )
    return remaining

def init(remaining, data_dict_list, wait_time):
    global counter
    exchange_bug = False
    iteration = 0

    # for _, row in tqdm(remaining.sort_values(by='conId').iloc[::-1].iterrows(), total=len(remaining)):
    # for _, row in tqdm(remaining.sort_values(by='conId').iterrows(),  total=len(remaining)):
    # for _, row in tqdm(remaining.iloc[counter::-1].iterrows(), total=len(remaining)):
    for _, row in tqdm(remaining.iloc[counter:].iterrows(), total=len(remaining)):
        ocr = PaddleOCR()
        profile, tradable, holding_types, dividends, top10, industries, countries, currencies, debtors, maturity, debt_type, fundamentals, lipper, style, funds_date = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        try:
            try:
                if exchange_bug:
                    search_exchange, search_symbol, exchange_bug = search_eft(ocr, row, 1)
                else:
                    search_exchange, search_symbol, exchange_bug = search_eft(ocr, row, wait_time)

                if search_symbol:
                    exact_search = True
                else:
                    exact_search = False
            except Exception as e:
                if e.args and len(e.args) > 0 and e.args[0] == 'PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen. To disable this fail-safe, set pyautogui.FAILSAFE to False. DISABLING FAIL-SAFE IS NOT RECOMMENDED.':
                    raise Exception('manual')
                exchange_bug = quick_search_etf(row)
                # if exchange_bug:
                #     exchange_bug = quick_search_etf(row, len(remaining['longName'].max()))
                # else:
                #     exchange_bug = quick_search_etf(row, 5)
                exact_search, search_symbol = False, None
                search_exchange = check_title(ocr, row['symbol'])
                if not search_exchange:
                    counter += 1
                    continue

            # Overview
            tradable = check_tradable(seconds=wait_time)
            try:
                style = extract_style()
            except Exception:
                continue
            profile = extract_profile(ocr)
            lipper = extract_lipper(ocr)
            holding_types = extract_holding_types(ocr)

            # Holdings tab
            if select_tab('holdings', wait_time):
                top10 = extract_top10(ocr)
                industries = extract_industry(ocr)
                countries = extract_country(ocr)
                currencies = extract_currency(ocr)

            # Bond data
            try:
                debtors = extract_debtors(ocr, 'debtor_quality')
                maturity = extract_debtors(ocr, 'maturity')
                debt_type = extract_debtors(ocr, 'debt_type')
            except Exception:
                pass

            # Ratios and Fundamentals tab
            if select_tab('fundamentals', wait_time):
                fundamentals, funds_date_str = extract_fundamentals(ocr)
                funds_date = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', funds_date_str).group(1)
                try:
                    funds_date = datetime.strptime(funds_date, '%m/%d/%Y').strftime('%Y-%m-%d')
                except ValueError as e:
                    print(f"Error parsing date '{funds_date}' from string '{funds_date_str}': {e}")

        except Exception as e:
            if exchange_bug or e.args and len(e.args) > 0 and e.args[0] == 'skip':
                pass
            elif e.args and len(e.args) > 0 and e.args[0] == 'PyAutoGUI fail-safe triggered from mouse moving to a corner of the screen. To disable this fail-safe, set pyautogui.FAILSAFE to False. DISABLING FAIL-SAFE IS NOT RECOMMENDED.':
                print(e)
                raise Exception('manual')
            elif e.args and len(e.args) > 0 and e.args[0] == 'manual':
                raise Exception('manual')
            else:
                traceback.print_exc()
                print(f'\nmain() {e} - Symbol: {row["symbol"]} - Name: {row["longName"]} - Exchange: {row["exchange"]}\n')
                counter += 1
                return

        data_dict = {
            'date_scraped': datetime.now().strftime('%Y-%m-%d'),
            'exchange_bug': exchange_bug,
            'exact_search': exact_search,
            'search_exchange': search_exchange,
            'search_symbol': search_symbol,
            'tradable': tradable,
            'profile': profile,
            'style': style,
            'lipper': lipper,
            'fundamentals': fundamentals,
            'funds_date': funds_date,
            'holding_types': holding_types,
            'top10': top10,
            'industries': industries,
            'countries': countries,
            'currencies': currencies,
            'debtors': debtors,
            'maturity': maturity,
            'debt_type': debt_type,
        }

        row_dict = row.to_dict()
        data_dict = {**row_dict, **data_dict}
        data_dict_list.append(data_dict)
        gc.collect()
        if exchange_bug:
            raise Exception(f'bug found')
        iteration += 1
        if iteration > 100:
            return


# MAIN
backup = pd.DataFrame()
counter = 0
while True:
    try:        
        switch_to_app()
        BOTTOM = 1070
        POSITIONS = {
            'file': (82, 44),
            'file_fund_option': (143, 120),
            'maximize': (51, 40),
            'search_box': (100, 45),
        }
        data_dict_list = []
        remaining = get_remaining()
        init(remaining, data_dict_list, wait_time=6)
        save(pd.DataFrame(data_dict_list))
        
    except Exception as e:
        print(counter)
        traceback.print_exc()
        if data_dict_list:
            save(pd.DataFrame(data_dict_list))
        else:
            raise Exception('none found')

        if e.args and len(e.args) > 0 and e.args[0] == 'bug found':
            print('bug found')
            break
        if e.args and len(e.args) > 0 and e.args[0] == 'manual':
            print('manual')
            break