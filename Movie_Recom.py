import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-medium"  # You can use "gpt2" or "gpt2-large" as well
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()

movie_recommendations = {
    "action": [
        "Mad Max: Fury Road", "John Wick", "Die Hard", "Gladiator", "The Dark Knight",
        "Mission: Impossible – Fallout", "The Matrix", "Terminator 2: Judgment Day", 
        "Lethal Weapon", "Speed", "War (Hindi)", "Shershaah (Hindi)", "Singham (Hindi)", 
        "Baahubali: The Beginning (Hindi)", "Dhoom 2 (Hindi)", "Ek Tha Tiger (Hindi)", 
        "Ghajini (Hindi)", "Commando (Hindi)", "Don (Hindi)", "Rowdy Rathore (Hindi)"
    ],
    "comedy": [
        "Superbad", "The Grand Budapest Hotel", "Groundhog Day", "Anchorman", 
        "Monty Python and the Holy Grail", "Ferris Bueller's Day Off", "Step Brothers", 
        "The Hangover", "Borat", "The 40-Year-Old Virgin", "Andaz Apna Apna (Hindi)", 
        "Chupke Chupke (Hindi)", "Gol Maal (1979, Hindi)", "Hera Pheri (Hindi)", 
        "3 Idiots (Hindi)", "Munna Bhai M.B.B.S. (Hindi)", "Dostana (Hindi)", 
        "Queen (Hindi)", "PK (Hindi)", "Bheja Fry (Hindi)"
    ],
    "drama": [
        "The Shawshank Redemption", "Forrest Gump", "Parasite", "The Godfather", 
        "Schindler's List", "A Beautiful Mind", "12 Years a Slave", "Fight Club", 
        "The Green Mile", "The Pursuit of Happyness", "Lagaan (Hindi)", "Taare Zameen Par (Hindi)", 
        "Swades (Hindi)", "Pink (Hindi)", "Barfi! (Hindi)", "Neerja (Hindi)", 
        "October (Hindi)", "The Lunchbox (Hindi)", "Article 15 (Hindi)", "Masaan (Hindi)"
    ],
    "sci-fi": [
        "Inception", "Blade Runner 2049", "The Matrix", "Interstellar", "2001: A Space Odyssey", 
        "Arrival", "The Martian", "Star Wars: Episode IV – A New Hope", "Avatar", 
        "Jurassic Park", "Koi... Mil Gaya (Hindi)", "PK (Hindi)", "Robot (Hindi)", 
        "Ra.One (Hindi)", "Cargo (Hindi)", "Tumbbad (Hindi)", "Ravan (Hindi)", 
        "Antariksham 9000 kmph (Hindi)", "Robot 2.0 (Hindi)", "Mr. India (Hindi)"
    ],
    "romance": [
        "Pride and Prejudice", "La La Land", "The Notebook", "Before Sunrise", 
        "Titanic", "A Walk to Remember", "Notting Hill", "500 Days of Summer", 
        "The Fault in Our Stars", "Eternal Sunshine of the Spotless Mind", 
        "Dilwale Dulhania Le Jayenge (Hindi)", "Kabir Singh (Hindi)", "Veer-Zaara (Hindi)", 
        "Jab We Met (Hindi)", "Ae Dil Hai Mushkil (Hindi)", "Kal Ho Naa Ho (Hindi)", 
        "Barfi! (Hindi)", "Lootera (Hindi)", "Rockstar (Hindi)", "Rehnaa Hai Terre Dil Mein (Hindi)"
    ],
    "horror": [
        "Hereditary", "A Quiet Place", "The Exorcist", "The Shining", "Get Out", 
        "The Conjuring", "Insidious", "It", "Paranormal Activity", "The Ring", 
        "Stree (Hindi)", "Tumbbad (Hindi)", "Pari (Hindi)", "Raaz (Hindi)", 
        "Bhool Bhulaiyaa (Hindi)", "Bhoot (Hindi)", "13B (Hindi)", 
        "Ek Thi Daayan (Hindi)", "Pizza (Hindi)", "Bulbbul (Hindi)"
    ]
}


def recommend_movies(genre):
    return movie_recommendations.get(genre.lower(), ["I'm not sure about that genre. Can you try another?"])

def generate_response(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def movie_chatbot():
    print("Hello! I'm your movie recommendation bot. Tell me about your favorite genre, and I'll suggest a movie!")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! Enjoy your movie!")
            break
        
        genre_found = False
        for genre in movie_recommendations.keys():
            if genre.lower() in user_input.lower():
                recommended_movies = recommend_movies(genre)
                bot_response = f"I recommend you watch {recommended_movies[0]}. Some other {genre} movies you might like are {', '.join(recommended_movies[1:])}."
                genre_found = True
                break
        
        if not genre_found:
            bot_response = generate_response(f"User: {user_input}\nBot:")

        print("Bot:", bot_response)

movie_chatbot()
