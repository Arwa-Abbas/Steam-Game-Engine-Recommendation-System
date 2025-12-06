import pandas as pd
import re
import sys
import os
from datetime import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.db import db
except ImportError:
    from db import db

def clean_price(price_str):
    """Extract numeric price from string"""
    if pd.isna(price_str):
        return 0.0
    price_str = str(price_str).replace("$", "").replace(",", "").strip()
    if price_str.lower() in ["free", ""]:
        return 0.0
    try:
        return float(price_str)
    except:
   
        return 0.0

def extract_number(review_str):
    """Extract total review count from string"""
    if pd.isna(review_str):
        return 0
    numbers = re.findall(r'\d+', str(review_str).replace(",", ""))
    return int(numbers[0]) if numbers else 0

def extract_sentiment_from_text(review_summary_text):
    """
    Extract sentiment score from review summary text
    Examples:
    - "Overwhelmingly Positive - 96% of the 128,900 user reviews" → 0.96
    - "Very Positive - 80% of the 701,597 user reviews" → 0.80
    - "Mixed - 65% positive" → 0.65
    """
    if pd.isna(review_summary_text) or not str(review_summary_text).strip():
        return 0.5  # Neutral default
    
    text = str(review_summary_text)
    

    # look for percentage pattern
    match = re.search(r'(\d+)%', text)
    if match:
        percentage = int(match.group(1))
        return percentage / 100  # Convert to 0-1 scale


    text_lower = text.lower()
    if 'overwhelmingly positive' in text_lower:
        return 0.95
    elif 'very positive' in text_lower:
        return 0.85
    elif 'positive' in text_lower:
        return 0.75
    elif 'mostly positive' in text_lower:
        return 0.70
    elif 'mixed' in text_lower:
        return 0.50
    elif 'mostly negative' in text_lower:
        return 0.35
    elif 'negative' in text_lower:
        return 0.25
    elif 'overwhelmingly negative' in text_lower:
        return 0.10
    
    return 0.5 


def get_sentiment_category(score):
    """Convert numeric score to category"""
    if score >= 0.95:
        return "overwhelmingly_positive"
    elif score >= 0.80:
        return "very_positive"
    elif score >= 0.70:
        return "mostly_positive"
    elif score >= 0.60:
        return "positive"
    elif score >= 0.40:
        return "mixed"
    elif score >= 0.30:
        return "mostly_negative"
    elif score >= 0.20:
        return "negative"
    else:
        return "overwhelmingly_negative"


def calculate_popularity_score(all_reviews_count, recent_reviews_count, 
                               all_sentiment_score, recent_sentiment_score):
    """
    Calculate popularity score (0-1) based on review counts and sentiment
    Higher review counts = more popular
    Better sentiment = quality boost
    """
    if all_reviews_count == 0:
        return 0.0
    
    # Review count score (logarithmic scale to handle wide ranges)
    # Games with 1M+ reviews should get close to 1.0
    import math
    max_reviews = 1000000  #
    review_score = min(1.0, math.log10(all_reviews_count + 1) / math.log10(max_reviews))
    

    # Recent activity bonus 
    if all_reviews_count > 0:
        recent_ratio = min(1.0, recent_reviews_count / all_reviews_count * 10)
        activity_bonus = recent_ratio * 0.1  # Up to 10% bonus
    else:
        activity_bonus = 0
    

    # Sentiment multiplier (quality factor)
    avg_sentiment = (all_sentiment_score + recent_sentiment_score) / 2
    sentiment_multiplier = 0.5 + (avg_sentiment * 0.5)  # 0.5 to 1.0 range
    
    # Final score
    base_popularity = review_score * sentiment_multiplier
    final_score = min(1.0, base_popularity + activity_bonus)
    
    return round(final_score, 3)


def clean_list_field(field):
    """Parse list field from CSV into Python list"""
    if pd.isna(field):
        return []
    if isinstance(field, str):
        field = field.strip()
        if field.startswith('[') and field.endswith(']'):
            field = field[1:-1]
        field = field.replace("'", "").replace('"', '')
        items = [x.strip().lower() for x in field.split(",") if x.strip()]
        return items
    return []


def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    return str(text).strip()

def extract_keywords_from_text(text, min_length=4):
    """
    Extract meaningful keywords from text
    Removes common words and keeps important game terms
    """
    if not text or pd.isna(text):
        return []
    

    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'for', 'with', 'your', 'you', 'are', 'can', 'will',
        'this', 'that', 'from', 'have', 'has', 'was', 'were', 'been',
        'their', 'they', 'them', 'there', 'what', 'when', 'where', 'which',
        'who', 'how', 'into', 'through', 'about', 'after', 'before', 'other'
    }
    
    # Clean and tokenize
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  
    words = text.split()


    # Filter keywords
    keywords = [
        word for word in words 
        if len(word) >= min_length 
        and word not in stop_words
        and not word.isdigit()
    ]


    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)
    
    return unique_keywords[:20]  


def parse_system_requirements(requirements_text):
    """
    Parse system requirements by splitting on | and extracting key specs
    Returns raw string format (not nested objects)
    """
    if pd.isna(requirements_text) or not str(requirements_text).strip():
        return ""
    
    text = str(requirements_text).strip()
    

    return text[:500]


def extract_specs_from_requirements(requirements_text):
    """
    Extract numeric specs from requirements string
    Returns dict with memory_gb, storage_gb, etc.
    """
    if pd.isna(requirements_text) or not str(requirements_text).strip():
        return {}
    
    text = str(requirements_text)
    specs = {}
    
    # Extract memory (RAM)
    memory_match = re.search(r'Memory:?\s*(\d+)\s*GB', text, re.IGNORECASE)
    if memory_match:
        specs["memory_gb"] = int(memory_match.group(1))
    
    # Extract storage
    storage_match = re.search(r'Storage:?\s*(\d+)\s*GB', text, re.IGNORECASE)
    if storage_match:
        specs["storage_gb"] = int(storage_match.group(1))
    
    # Extract VRAM
    vram_match = re.search(r'(\d+)\s*GB\+?\s*(?:of\s+)?VRAM', text, re.IGNORECASE)
    if vram_match:
        specs["vram_gb"] = int(vram_match.group(1))
    
    # Extract DirectX version
    dx_match = re.search(r'DirectX:?\s*Version\s*(\d+)', text, re.IGNORECASE)
    if dx_match:
        specs["directx_version"] = int(dx_match.group(1))
    
    # Check for SSD requirement
    if re.search(r'\bSSD\b', text, re.IGNORECASE):
        specs["ssd_required"] = True
    
    # Extract OS type
    if 'Windows' in text:
        specs["os_type"] = "windows"
        # Extract Windows version
        win_match = re.search(r'Windows\s+(\d+)', text, re.IGNORECASE)
        if win_match:
            specs["os_version"] = int(win_match.group(1))
        # Check for 64-bit
        if '64-bit' in text:
            specs["architecture"] = "64-bit"
    elif 'Mac' in text or 'macOS' in text:
        specs["os_type"] = "mac"
    elif 'Linux' in text:
        specs["os_type"] = "linux"
    

    # Extract GPU brand
    if 'nvidia' in text.lower() or 'gtx' in text.lower() or 'rtx' in text.lower():
        specs["gpu_brand"] = "nvidia"
    elif 'amd' in text.lower() or 'radeon' in text.lower():
        specs["gpu_brand"] = "amd"
    elif 'intel' in text.lower() and 'graphics' in text.lower():
        specs["gpu_brand"] = "intel"
    

    # Extract CPU brand
    if 'intel' in text.lower() and ('processor' in text.lower() or 'cpu' in text.lower()):
        specs["cpu_brand"] = "intel"
    elif 'amd' in text.lower() and ('processor' in text.lower() or 'cpu' in text.lower() or 'fx' in text.lower() or 'ryzen' in text.lower()):
        specs["cpu_brand"] = "amd"
    
    return specs


def categorize_game(tags, features, price, description):
    """
    Create high-level categories for the game
    """
    categories = []
    
    # Genre categories from tags
    genre_mapping = {
        'action': ['action', 'fps', 'shooter', 'platformer', 'hack and slash'],
        'rpg': ['rpg', 'jrpg', 'crpg', 'roguelike', 'roguelite'],
        'strategy': ['strategy', 'rts', 'turn-based', 'tower defense', 'grand strategy'],
        'adventure': ['adventure', 'exploration', 'walking simulator', 'narrative'],
        'simulation': ['simulation', 'sim', 'management', 'building', 'city builder'],
        'sports': ['sports', 'racing', 'football', 'soccer', 'basketball'],
        'puzzle': ['puzzle', 'logic', 'match'],
        'horror': ['horror', 'survival horror', 'psychological horror'],
        'indie': ['indie', 'casual']
    }
    
    tags_lower = [t.lower() for t in tags]
    for category, keywords in genre_mapping.items():
        if any(keyword in tag for tag in tags_lower for keyword in keywords):
            categories.append(category)
    
    # Multiplayer category
    multiplayer_keywords = ['multiplayer', 'co-op', 'online', 'pvp', 'mmo']
    if any(kw in tag for tag in tags_lower for kw in multiplayer_keywords):
        categories.append('multiplayer')
    
    if 'single-player' in ' '.join(features).lower():
        categories.append('singleplayer')
    
    # Price category
    if price == 0:
        categories.append('free_to_play')
    elif price < 10:
        categories.append('budget')
    elif price < 30:
        categories.append('mid_price')
    else:
        categories.append('premium')
    
    return list(set(categories))  


def extract_year(date_str):
    """Extract year from date string"""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str)
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return int(year_match.group(0))
    return None


def load_records(csv_path="dataset/steam_game_engine.csv", append=False):
   
    try:
        print(f"Loading from: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}")
            return 0
        
        # Drop collection if not appending
        if not append and "steam_games" in db.list_collection_names():
            count_before = db.steam_games.count_documents({})
            db.steam_games.drop()
            print(f"Cleared {count_before} old records")
        elif append:
            print(f"Appending to existing data")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
        
        records = []
        
        print(f"\ Processing records...\n")
        
        for index, row in df.iterrows():
            try:
                # Extract basic info
                title = clean_text(row.get("Title", ""))
                original_price = clean_price(row.get("Original Price"))
                discounted_price = clean_price(row.get("Discounted Price"))
                release_date = clean_text(row.get("Release Date", ""))
                release_year = extract_year(release_date)
                description = clean_text(row.get("Game Description", ""))
                
                # Extract review data
                recent_review_text = clean_text(row.get("Recent Reviews Summary", ""))
                all_review_text = clean_text(row.get("All Reviews Summary", ""))
                recent_reviews_count = extract_number(row.get("Recent Reviews Number", 0))
                all_reviews_count = extract_number(row.get("All Reviews Number", 0))
                
                # Extract sentiment scores from review text
                recent_sentiment_score = extract_sentiment_from_text(recent_review_text)
                all_sentiment_score = extract_sentiment_from_text(all_review_text)
                
                # Get sentiment categories
                recent_sentiment_category = get_sentiment_category(recent_sentiment_score)
                all_sentiment_category = get_sentiment_category(all_sentiment_score)
                
                # Parse lists
                tags = clean_list_field(row.get("Popular Tags", []))
                features = clean_list_field(row.get("Game Features", []))
                languages = clean_list_field(row.get("Supported Languages", []))
                
                # Extract keywords from description
                description_keywords = extract_keywords_from_text(description)
                
                # Combine all keywords for search
                all_keywords = list(set(tags + description_keywords))
                
                # Parse system requirements (raw string + extracted specs)
                requirements_text = row.get("Minimum Requirements", "")
                minimum_requirements = parse_system_requirements(requirements_text)
                extracted_specs = extract_specs_from_requirements(requirements_text)
                
                # Calculate popularity score based on review counts and sentiment
                popularity = calculate_popularity_score(
                    all_reviews_count,
                    recent_reviews_count,
                    all_sentiment_score,
                    recent_sentiment_score
                )
                
                # Categorize game
                categories = categorize_game(tags, features, original_price, description)
                
                # Calculate overall sentiment
                overall_sentiment_score = round((recent_sentiment_score + all_sentiment_score) / 2, 2)
                overall_sentiment_category = get_sentiment_category(overall_sentiment_score)
                
                # Create record
                record = {
                    # Basic Info
                    "title": title,
                    "title_lower": title.lower(),
                    "original_price": original_price,
                    "discounted_price": discounted_price,
                    "discount_percentage": round((1 - discounted_price/original_price) * 100, 1) if original_price > 0 else 0,
                    "release_date": release_date,
                    "release_year": release_year,
                    "link": clean_text(row.get("Link", "")),
                    "description": description,
                    
                    # Reviews (separate fields, not nested)
                    "recent_reviews_summary": recent_review_text,
                    "recent_reviews_count": recent_reviews_count,
                    "recent_sentiment_score": recent_sentiment_score,
                    "recent_sentiment_category": recent_sentiment_category,
                    
                    "all_reviews_summary": all_review_text,
                    "all_reviews_count": all_reviews_count,
                    "all_sentiment_score": all_sentiment_score,
                    "all_sentiment_category": all_sentiment_category,
                    
                    # Overall sentiment
                    "overall_sentiment_score": overall_sentiment_score,
                    "overall_sentiment_category": overall_sentiment_category,
                    
                    # Company
                    "developer": clean_text(row.get("Developer", "")).lower(),
                    "publisher": clean_text(row.get("Publisher", "")).lower(),
                    
                    # Tags & Features
                    "tags": tags,
                    "features": features,
                    "languages": languages,
                    "categories": categories,
                    
                    # Keywords for search/recommendation
                    "keywords": all_keywords,
                    "description_keywords": description_keywords,
                    
                    # System Requirements (raw string)
                    "minimum_requirements": minimum_requirements,
                    
                    # Extracted specs (flat structure)
                    "memory_gb": extracted_specs.get("memory_gb"),
                    "storage_gb": extracted_specs.get("storage_gb"),
                    "vram_gb": extracted_specs.get("vram_gb"),
                    "directx_version": extracted_specs.get("directx_version"),
                    "ssd_required": extracted_specs.get("ssd_required", False),
                    "os_type": extracted_specs.get("os_type"),
                    "os_version": extracted_specs.get("os_version"),
                    "architecture": extracted_specs.get("architecture"),
                    "gpu_brand": extracted_specs.get("gpu_brand"),
                    "cpu_brand": extracted_specs.get("cpu_brand"),
                    
                    # Scores
                    "popularity_score": popularity,
                    
                    # Price category
                    "price_category": "free" if original_price == 0 else (
                        "budget" if original_price < 10 else (
                            "mid_price" if original_price < 30 else "premium"
                        )
                    ),
                    
                    # Metadata
                    "indexed_at": datetime.now()
                }
                
                records.append(record)
                
                # Show progress with sample
                if (index + 1) % 25 == 0:
                    print(f"✓ Processed {index + 1}/100 records")
                    # Show sample
                    if index == 24:
                        print(f"\n   Sample: {title[:40]}")
                        print(f"   Reviews: {all_reviews_count:,} total | Sentiment: {all_sentiment_score:.2f} ({all_sentiment_category})")
                        print(f"   Popularity: {popularity:.3f}\n")
                
            except Exception as e:
                print(f"Skipping row {index} ({row.get('Title', 'Unknown')}): {str(e)[:60]}")
                continue
        
        # Insert to MongoDB
        if records:
            db.steam_games.insert_many(records)
            
            # Create indexes
            print(f"\nCreating indexes...")
            db.steam_games.create_index([("title_lower", 1)])
            db.steam_games.create_index([("tags", 1)])
            db.steam_games.create_index([("keywords", 1)])
            db.steam_games.create_index([("popularity_score", -1)])
            db.steam_games.create_index([("overall_sentiment_score", -1)])
            db.steam_games.create_index([("original_price", 1)])
            db.steam_games.create_index([("categories", 1)])
            db.steam_games.create_index([("all_reviews_count", -1)])
            
            print(f"\n{'='*60}")
            print(f"Successfully loaded {len(records)} records!")
            print(f"{'='*60}")
            
            # Show statistics
            print(f"\nDataset Statistics:")
            print(f"   • Total games: {len(records)}")
            print(f"   • Free games: {sum(1 for r in records if r['original_price'] == 0)}")
            print(f"   • Paid games: {sum(1 for r in records if r['original_price'] > 0)}")
            print(f"   • Avg price: ${sum(r['original_price'] for r in records) / len(records):.2f}")
            
            # Popularity stats
            pop_scores = [r['popularity_score'] for r in records]
            print(f"   • Popularity range: {min(pop_scores):.3f} - {max(pop_scores):.3f}")
            print(f"   • Avg popularity: {sum(pop_scores) / len(pop_scores):.3f}")
            
            # Sentiment stats
            sentiment_scores = [r['overall_sentiment_score'] for r in records]
            print(f"   • Sentiment range: {min(sentiment_scores):.2f} - {max(sentiment_scores):.2f}")
            print(f"   • Avg sentiment: {sum(sentiment_scores) / len(sentiment_scores):.2f}")
            
            # Show top games by popularity
            print(f"\nTop 3 Most Popular Games:\n")
            top_games = sorted(records, key=lambda x: x['popularity_score'], reverse=True)[:3]
            
            for i, game in enumerate(top_games, 1):
                print(f"{i}. {game['title']}")
                print(f"   Price: ${game['original_price']:.2f}")
                print(f"   Reviews: {game['all_reviews_count']:,} | Sentiment: {game['all_sentiment_score']:.2f} ({game['all_sentiment_category']})")
                print(f"   Popularity: {game['popularity_score']:.3f}")
                print(f"   Tags: {', '.join(game['tags'][:5])}")
                if game.get('memory_gb'):
                    print(f"   RAM: {game['memory_gb']}GB", end="")
                if game.get('storage_gb'):
                    print(f" | Storage: {game['storage_gb']}GB", end="")
                print("\n")
            
            return len(records)
        
        return 0
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    print(" Steam Game Data Loader")
    print("=" * 60)
   
    count = load_records(
        csv_path="dataset/steam_game_engine.csv", 
        append=False
    )
    
    if count > 0:
        print(f"\n Successfully loaded {count} games into MongoDB!")
        print(f"💡 Database ready for knowledge-based recommendations")
    else:
        print(f"\n No records were loaded")
    
    print("\nLoader completed!")