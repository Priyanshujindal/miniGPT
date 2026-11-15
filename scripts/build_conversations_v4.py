import json
import random
from pathlib import Path
from tokenizers import Tokenizer

from training.config_v4 import (
    TOKENIZER_V4_PATH,
    DATA_DIR_V4,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    DATASET_REPORT_PATH,
    BLOCK_SIZE,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
    END_TOKEN,
)

SEED = 42
random.seed(SEED)


def format_conversation(messages):
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "system":
            formatted += f"{SYSTEM_TOKEN}{content}{END_OF_TURN_TOKEN}"
        elif role == "user":
            formatted += f"{USER_TOKEN}{content}{END_OF_TURN_TOKEN}"
        elif role == "assistant":
            formatted += f"{ASSISTANT_TOKEN}{content}{END_OF_TURN_TOKEN}"
    formatted += END_TOKEN
    return formatted


# Rich, multi-category conversational bank with 100% UNIQUE user queries
RAW_ITEMS = [
    # ================= 1. General Knowledge =================
    ("general_knowledge", "What is the capital of France?", "The capital of France is Paris."),
    ("general_knowledge", "What planet is closest to the Sun?", "Mercury is the closest planet to the Sun."),
    ("general_knowledge", "How many continents are there on Earth?", "There are seven continents on Earth: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."),
    ("general_knowledge", "What is the largest ocean on Earth?", "The Pacific Ocean is the largest and deepest ocean on Earth."),
    ("general_knowledge", "What gas do humans need to breathe to survive?", "Humans need oxygen to breathe and survive."),
    ("general_knowledge", "What is the chemical formula for water?", "The chemical formula for water is H2O, meaning two hydrogen atoms bonded to one oxygen atom."),
    ("general_knowledge", "What is the tallest mountain on Earth?", "Mount Everest is the highest mountain above sea level, located in the Himalayas."),
    ("general_knowledge", "What color do you get when you mix red and yellow?", "Mixing red and yellow paint creates orange."),
    ("general_knowledge", "How many days are in a standard year?", "A standard year has 365 days, while a leap year has 366 days."),
    ("general_knowledge", "Which flightless bird lives in Antarctica?", "The penguin is a flightless bird well adapted to living in Antarctica."),
    ("general_knowledge", "What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("general_knowledge", "What is the capital of Italy?", "The capital of Italy is Rome."),
    ("general_knowledge", "What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("general_knowledge", "What is the capital of Canada?", "The capital of Canada is Ottawa."),
    ("general_knowledge", "What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("general_knowledge", "What force keeps objects on the ground?", "Gravity is the natural force that pulls objects toward the center of the Earth."),
    ("general_knowledge", "What is the hardest natural mineral?", "Diamond is the hardest natural mineral known on Earth."),
    ("general_knowledge", "Which animal is the largest mammal on Earth?", "The blue whale is the largest mammal and the largest animal on Earth."),
    ("general_knowledge", "What do bees produce?", "Bees produce honey by gathering nectar from flowering plants."),
    ("general_knowledge", "What is the boiling point of water in Celsius?", "Water boils at 100 degrees Celsius under standard atmospheric pressure."),
    ("general_knowledge", "What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("general_knowledge", "Which continent is Egypt located on?", "Egypt is located on the continent of Africa."),
    ("general_knowledge", "What natural satellite orbits the Earth?", "The Moon is the natural satellite that orbits planet Earth."),
    ("general_knowledge", "What color are emeralds?", "Emeralds are precious gemstones that are green in color."),
    ("general_knowledge", "How many sides does an octagon have?", "An octagon has eight sides."),
    ("general_knowledge", "How many sides does a triangle have?", "A triangle has three sides."),
    ("general_knowledge", "How many sides does a hexagon have?", "A hexagon has six sides."),
    ("general_knowledge", "What is the primary language spoken in Spain?", "Spanish is the primary language spoken in Spain."),
    ("general_knowledge", "What is the largest desert in the world?", "The Antarctic Desert is the largest cold desert, and the Sahara is the largest hot desert."),
    ("general_knowledge", "What organ pumps blood throughout the human body?", "The heart pumps blood through the circulatory system to nourish the body."),
    ("general_knowledge", "Which planet is famous for its prominent rings?", "Saturn is the planet most famous for its extensive and bright ring system."),
    ("general_knowledge", "What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("general_knowledge", "What is the currency of the United States?", "The currency of the United States is the US Dollar."),
    ("general_knowledge", "What is the largest land mammal?", "The African elephant is the largest living land mammal."),
    ("general_knowledge", "Which organ is responsible for breathing in humans?", "The lungs are the primary organs of the human respiratory system."),
    ("general_knowledge", "What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("general_knowledge", "What color is a ripe banana?", "A ripe banana is typically yellow."),
    ("general_knowledge", "How many hours are there in one day?", "There are 24 hours in a single day."),
    ("general_knowledge", "What is the capital of India?", "The capital of India is New Delhi."),
    ("general_knowledge", "Which ocean is located between the Americas and Europe?", "The Atlantic Ocean lies between the Americas and Europe and Africa."),

    # ================= 2. Education & Science =================
    ("education", "What is photosynthesis?", "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."),
    ("education", "What is an ecosystem?", "An ecosystem is a biological community of interacting living organisms and their physical environment."),
    ("education", "Why is the sky blue during the day?", "The sky looks blue because the gases in Earth's atmosphere scatter sunlight in all directions, and blue light scatters more than other colors because it travels as shorter waves."),
    ("education", "What is an atom?", "An atom is the basic building block of matter, composed of protons, neutrons, and orbiting electrons."),
    ("education", "What is the difference between a herbivore and a carnivore?", "A herbivore eats only plants, while a carnivore feeds primarily on the meat of other animals."),
    ("education", "What is condensation?", "Condensation is the physical change of matter from a gas phase into a liquid phase, such as water vapor turning into water droplets."),
    ("education", "What is evaporation?", "Evaporation is the process where liquid water absorbs heat and transforms into water vapor gas."),
    ("education", "Why do leaves change color in the fall?", "As daylight hours decrease in autumn, trees stop producing green chlorophyll, which unmasks the yellow, orange, and red pigments in leaves."),
    ("education", "What is gravity?", "Gravity is a fundamental force of attraction between any two objects that have mass."),
    ("education", "What causes earthquakes?", "Earthquakes are caused by a sudden release of energy in Earth's crust that creates seismic waves, usually along tectonic fault lines."),
    ("education", "What is renewable energy?", "Renewable energy comes from natural sources that are replenished faster than they are consumed, like solar, wind, and water power."),
    ("education", "What is a solar eclipse?", "A solar eclipse occurs when the Moon passes directly between the Earth and the Sun, temporarily blocking the Sun's light."),
    ("education", "What is a lunar eclipse?", "A lunar eclipse occurs when the Earth passes between the Sun and the Moon, casting Earth's shadow across the Moon."),
    ("education", "What is friction in physics?", "Friction is a force between surfaces in contact that resists their relative motion."),
    ("education", "What is the water cycle?", "The water cycle describes how water continuously evaporates into the air, condenses into clouds, and precipitates back as rain or snow."),
    ("education", "What is a mammal?", "A mammal is a warm-blooded vertebrate animal distinguished by hair or fur and females producing milk for their young."),
    ("education", "What is velocity?", "Velocity is the speed of an object in a specific direction."),
    ("education", "What is an endangered species?", "An endangered species is an animal or plant species that is at serious risk of extinction."),
    ("education", "What is an omnivore?", "An omnivore is an organism that regularly consumes both plant and animal matter."),
    ("education", "What is kinetic energy?", "Kinetic energy is the energy possessed by an object due to its motion."),
    ("education", "What is potential energy?", "Potential energy is stored energy stored in an object due to its position or state."),
    ("education", "What is biodiversity?", "Biodiversity refers to the variety of life on Earth across all species, genetics, and ecosystems."),
    ("education", "What is climate change?", "Climate change refers to long-term shifts in temperatures and weather patterns across Earth."),
    ("education", "What is an invertebrate?", "An invertebrate is an animal that does not possess or develop a backbone or vertebral column."),

    # ================= 3. Programming & Python =================
    ("programming", "How do I print text in Python?", "Use the print() function: print('Hello, world!')"),
    ("programming", "What is a variable in Python?", "A variable is a named storage container in memory used to store data values like numbers or text."),
    ("programming", "How do you define a function in Python?", "You define a function using the def keyword:\ndef greet(name):\n    return 'Hello, ' + name"),
    ("programming", "How do you create a list in Python?", "You create a list with square brackets: fruits = ['apple', 'banana', 'cherry']"),
    ("programming", "What is a boolean data type?", "A boolean is a data type with only two possible values: True or False."),
    ("programming", "What does a for loop do in Python?", "A for loop iterates over a sequence like a list, string, or range, executing code for each item."),
    ("programming", "What is an if-else statement?", "An if-else statement allows a program to execute different blocks of code depending on whether a condition is true or false."),
    ("programming", "How do you write a single-line comment in Python?", "Start the line with a hash symbol (#): # This is a comment."),
    ("programming", "What is a string in Python?", "A string is a sequence of characters enclosed in single or double quotes, such as 'hello' or \"python\"."),
    ("programming", "What does the len() function do?", "The len() function returns the total number of items in a sequence like a string or list."),
    ("programming", "What is a bug in computer science?", "A bug is an unexpected error, flaw, or failure in a computer program that prevents it from working correctly."),
    ("programming", "What is debugging?", "Debugging is the routine process of finding and removing bugs, errors, or anomalies from computer software."),
    ("programming", "How do you append an item to a list in Python?", "Use the .append() method: my_list.append(item)"),
    ("programming", "What is a dictionary in Python?", "A dictionary is an unordered collection of key-value pairs created with curly braces: user = {'name': 'Alice', 'age': 25}"),
    ("programming", "What does the range() function produce in Python?", "range(n) produces a sequence of numbers starting from 0 up to n-1."),
    ("programming", "How do you check equality in Python?", "Use the double equals operator (==): if x == 10: print('Match')"),
    ("programming", "What is a syntax error in Python?", "A syntax error occurs when code violates the grammar rules of the Python language, preventing it from parsing."),
    ("programming", "What is the difference between = and == in Python?", "The single equals (=) assigns a value to a variable, while the double equals (==) tests whether two values are equal."),
    ("programming", "What does str.lower() do in Python?", "The .lower() method converts all uppercase characters in a string to lowercase."),
    ("programming", "What does str.upper() do in Python?", "The .upper() method converts all lowercase characters in a string to uppercase."),
    ("programming", "What is a float in Python?", "A float is a data type used to represent real numbers that have a decimal point, such as 3.14 or -0.5."),
    ("programming", "What does the type() function do in Python?", "The type() function returns the class data type of an object, like <class 'int'> or <class 'str'>."),
    ("programming", "What is a while loop in Python?", "A while loop repeatedly executes a block of code as long as its condition remains True."),
    ("programming", "How do you convert a string '10' into an integer in Python?", "Use int(): number = int('10')"),
    ("programming", "What is an index in a Python list?", "An index is an integer position of an element in a list, starting at 0 for the first element."),

    # ================= 4. Instructions & Formatting =================
    ("instructions", "List three healthy daily habits.", "1. Drink plenty of fresh water.\n2. Exercise for at least 30 minutes.\n3. Sleep 7 to 8 hours each night."),
    ("instructions", "Write a short thank you note to a friend.", "Dear friend, thank you so much for your help and kindness today! I truly appreciate your support."),
    ("instructions", "Rewrite this to be polite: 'Give me the report now.'", "Could you please send me the report when you get a chance? Thank you!"),
    ("instructions", "List three common citrus fruits.", "1. Orange\n2. Lemon\n3. Lime"),
    ("instructions", "Give simple steps to wash your hands properly.", "1. Wet hands with clean water.\n2. Apply soap and lather thoroughly for 20 seconds.\n3. Rinse well and dry with a clean towel."),
    ("instructions", "Format these names in alphabetical order: Sarah, David, Alex.", "1. Alex\n2. David\n3. Sarah"),
    ("instructions", "Give three safety tips for crossing the street.", "1. Look left, right, and left again.\n2. Always use a designated pedestrian crosswalk.\n3. Make eye contact with drivers before crossing."),
    ("instructions", "List four seasons of the year in order.", "1. Spring\n2. Summer\n3. Autumn (Fall)\n4. Winter"),
    ("instructions", "List three primary colors.", "The three primary colors of light pigment are red, yellow, and blue."),
    ("instructions", "Convert this text into bullet points: Apples are red. Bananas are yellow. Grapes are purple.", "- Apples are red.\n- Bananas are yellow.\n- Grapes are purple."),
    ("instructions", "List three things plants need to grow.", "1. Sunlight\n2. Water\n3. Nutrient-rich soil"),
    ("instructions", "Give two tips for studying effectively.", "1. Study in short, focused 25-minute intervals.\n2. Test yourself using practice questions instead of passive reading."),
    ("instructions", "Alphabetize these animals: Zebra, Cat, Monkey.", "1. Cat\n2. Monkey\n3. Zebra"),
    ("instructions", "Rewrite this informal text professionally: 'Yeah I can do that tmrw.'", "Yes, I will be able to complete that task tomorrow."),
    ("instructions", "List three safe kitchen practices.", "1. Wash cutting boards after handling raw meat.\n2. Keep knife handles dry and secure.\n3. Clean spills immediately to avoid slipping."),

    # ================= 5. Everyday Conversation =================
    ("everyday_conversation", "Hello! How are you today?", "Hello! I am doing well, thank you. How can I assist you today?"),
    ("everyday_conversation", "Good morning!", "Good morning! I hope you have a productive and great day ahead."),
    ("everyday_conversation", "Good afternoon!", "Good afternoon! How can I help you today?"),
    ("everyday_conversation", "Good evening!", "Good evening! What can I help you with tonight?"),
    ("everyday_conversation", "Thank you for the help!", "You are very welcome! Feel free to ask if you need anything else."),
    ("everyday_conversation", "Who created you?", "I am MiniGPT V4, an instruction-tuned language model trained using PyTorch."),
    ("everyday_conversation", "What is your purpose?", "My purpose is to assist users by answering questions, explaining concepts, solving simple math, and writing code."),
    ("everyday_conversation", "Can you help me with a question?", "Of course! Ask me whatever you would like to know."),
    ("everyday_conversation", "Goodbye!", "Goodbye! Take care and have a wonderful day!"),
    ("everyday_conversation", "Nice to meet you!", "Nice to meet you as well! What would you like to explore today?"),
    ("everyday_conversation", "I am feeling happy today.", "That is wonderful to hear! I hope your day stays positive and joyful."),
    ("everyday_conversation", "Are you an AI?", "Yes, I am a compact AI language model designed for conversational assistance."),
    ("everyday_conversation", "What languages can you program in?", "I know Python best, but I understand basic concepts across multiple programming languages."),
    ("everyday_conversation", "Tell me something interesting.", "Did you know that honey never spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are thousands of years old and still edible!"),
    ("everyday_conversation", "What is your favorite color?", "I do not see colors like humans, but I think green is wonderful because it symbolizes nature and growth!"),
    ("everyday_conversation", "How does your day look?", "My day is great! I am ready to help you with any questions or tasks you have."),

    # ================= 6. Multi-Turn Conversations =================
    ("multi_turn", [
        {"role": "user", "content": "I have a puppy named Milo."},
        {"role": "assistant", "content": "Milo is a great name! What breed is he?"},
        {"role": "user", "content": "He is a golden retriever."},
        {"role": "assistant", "content": "Golden retrievers are wonderful companions! They are friendly, playful, and very loyal."}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "What is 12 plus 18?"},
        {"role": "assistant", "content": "12 plus 18 equals 30."},
        {"role": "user", "content": "Now multiply that answer by 2."},
        {"role": "assistant", "content": "30 multiplied by 2 equals 60."}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "I want to make a fruit salad."},
        {"role": "assistant", "content": "Fruit salad is refreshing! What fruits do you plan to use?"},
        {"role": "user", "content": "Strawberries, blueberries, and bananas."},
        {"role": "assistant", "content": "That is a delicious combination! Chop the bananas and strawberries, toss gently with blueberries, and enjoy."}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "Can you recommend a beginner programming language?"},
        {"role": "assistant", "content": "I recommend Python. Its syntax reads almost like English, making it ideal for beginners."},
        {"role": "user", "content": "What can I build with it?"},
        {"role": "assistant", "content": "You can build websites, automated scripts, games, data analysis tools, and machine learning models with Python!"}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "Let us count by tens up to forty."},
        {"role": "assistant", "content": "Sure! 10, 20, 30, 40."},
        {"role": "user", "content": "What is the next number?"},
        {"role": "assistant", "content": "The next number counting by tens is 50."}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "What is the capital of Spain?"},
        {"role": "assistant", "content": "The capital of Spain is Madrid."},
        {"role": "user", "content": "What language is spoken there?"},
        {"role": "assistant", "content": "Spanish (Castilian) is the official and primary language spoken in Madrid."}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "I am planning a road trip."},
        {"role": "assistant", "content": "Road trips are exciting! Where are you thinking of going?"},
        {"role": "user", "content": "To the mountains for camping."},
        {"role": "assistant", "content": "Mountains are fantastic for camping! Remember to pack warm clothes, a tent, plenty of water, and a first-aid kit."}
    ]),
    ("multi_turn", [
        {"role": "user", "content": "How do I define a function in Python?"},
        {"role": "assistant", "content": "Use the def keyword followed by the function name, parentheses, and a colon:\ndef my_function():\n    pass"},
        {"role": "user", "content": "How do I make it take an argument?"},
        {"role": "assistant", "content": "Place the argument name inside the parentheses:\ndef greet(name):\n    print('Hello, ' + name)"}
    ]),

    # ================= 7. Summarization =================
    ("summarization", "Summarize this: The library was quiet and peaceful. Maya found an ancient book about castle legends and read it until sunset.", "Summary: Maya spent a peaceful afternoon in a library reading a book about castle legends."),
    ("summarization", "Summarize: Leo found a lost puppy shivering under a bench in the pouring rain. He took it home, wrapped it in a towel, and gave it food.", "Summary: Leo rescued a cold, lost puppy from the rain and cared for it at home."),
    ("summarization", "Summarize: Solar panels convert sunlight directly into electricity, providing a renewable energy source that reduces greenhouse gases.", "Summary: Solar panels turn sunlight into clean, renewable electricity to cut emissions."),
    ("summarization", "Summarize: During winter, brown bears enter hibernation, lowering their heart rate and body temperature to survive months without food.", "Summary: Brown bears hibernate in winter to conserve energy when food is scarce."),
    ("summarization", "Summarize: In 1969, Apollo 11 landed astronauts Neil Armstrong and Buzz Aldrin on the Moon, marking a historic achievement for human space exploration.", "Summary: In 1969, the Apollo 11 mission achieved humanity's first Moon landing."),
    ("summarization", "Summarize: Honeybees visit flowers to collect nectar and pollen, accidentally pollinating crops that humans rely on for food.", "Summary: Bees pollinate vital human food crops while gathering nectar and pollen from flowers."),
    ("summarization", "Summarize: Regular exercise strengthens muscles, improves heart health, and releases endorphins that boost everyday mental well-being.", "Summary: Regular exercise boosts physical cardiovascular health and enhances mood."),
    ("summarization", "Summarize: Rain fell steadily all morning, watering the dry farmland and filling the irrigation canals for the local farmers.", "Summary: Morning rain replenished the dry farmland and filled essential irrigation canals."),
    ("summarization", "Summarize: Coral reefs provide habitat for over 25 percent of marine species despite covering less than one percent of the ocean floor.", "Summary: Coral reefs shelter a quarter of marine life despite their small ocean footprint."),
    ("summarization", "Summarize: Electric cars use rechargeable battery packs instead of gasoline engines, producing zero tailpipe emissions while driving.", "Summary: Electric vehicles run on rechargeable batteries, eliminating tailpipe emissions."),
]


def build_unique_dataset():
    unique_samples = []
    seen_user_queries = set()

    # 1. Base samples
    for item in RAW_ITEMS:
        category = item[0]
        if isinstance(item[1], list):
            messages = item[1]
            user_key = messages[0]["content"].strip().lower()
        else:
            messages = [
                {"role": "user", "content": item[1]},
                {"role": "assistant", "content": item[2]}
            ]
            user_key = item[1].strip().lower()

        if user_key in seen_user_queries:
            continue
        seen_user_queries.add(user_key)
        unique_samples.append({"category": category, "messages": messages})

    # 2. Distinct Arithmetic questions (each with unique numbers, no duplicates)
    # Controlled set of 120 unique arithmetic questions
    for a in range(2, 22):
        for b in range(2, 8):
            q_add = f"What is {a} plus {b}?"
            if q_add.lower() not in seen_user_queries:
                seen_user_queries.add(q_add.lower())
                unique_samples.append({
                    "category": "mathematics",
                    "messages": [
                        {"role": "user", "content": q_add},
                        {"role": "assistant", "content": f"{a} plus {b} equals {a + b}."}
                    ]
                })

    for a in range(25, 65, 3):
        for b in range(3, 15, 2):
            q_sub = f"Calculate {a} minus {b}."
            if q_sub.lower() not in seen_user_queries:
                seen_user_queries.add(q_sub.lower())
                unique_samples.append({
                    "category": "mathematics",
                    "messages": [
                        {"role": "user", "content": q_sub},
                        {"role": "assistant", "content": f"{a} minus {b} is {a - b}."}
                    ]
                })

    for a in range(2, 10):
        for b in range(2, 10):
            q_mul = f"What is {a} multiplied by {b}?"
            if q_mul.lower() not in seen_user_queries:
                seen_user_queries.add(q_mul.lower())
                unique_samples.append({
                    "category": "mathematics",
                    "messages": [
                        {"role": "user", "content": q_mul},
                        {"role": "assistant", "content": f"{a} multiplied by {b} is {a * b}."}
                    ]
                })

    # 3. Programming syntax queries (distinct values)
    data_vals = [("x", 42), ("y", 100), ("total", 500), ("score", 10), ("count", 5), ("price", 99)]
    for var, val in data_vals:
        q_var = f"How do I set a Python variable named {var} to {val}?"
        if q_var.lower() not in seen_user_queries:
            seen_user_queries.add(q_var.lower())
            unique_samples.append({
                "category": "programming",
                "messages": [
                    {"role": "user", "content": q_var},
                    {"role": "assistant", "content": f"{var} = {val}"}
                ]
            })

    return unique_samples


def main():
    print("=" * 70)
    print("BUILDING ZERO-LEAKAGE MINIGPT V4 CONVERSATION DATASET")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_V4_PATH))
    raw_candidates = build_unique_dataset()
    print(f"\nGenerated {len(raw_candidates)} distinct candidate records.")

    # Validate and tokenize
    valid_samples = []
    category_counts = {}

    for idx, item in enumerate(raw_candidates):
        msgs = item["messages"]
        cat = item["category"]

        formatted = format_conversation(msgs)
        encoded = tokenizer.encode(formatted)
        length = len(encoded.ids)

        if length > BLOCK_SIZE:
            continue

        valid_samples.append({
            "id": f"v4_{idx:05d}",
            "category": cat,
            "messages": msgs,
            "token_count": length,
            "user_prompt": msgs[0]["content"].strip().lower() if msgs[0]["role"] == "user" else msgs[1]["content"].strip().lower()
        })
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"Validated {len(valid_samples)} samples within {BLOCK_SIZE}-token limit.")
    print("\nCategory Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / len(valid_samples) * 100
        print(f"  {cat:25s}: {count:4d} ({pct:5.1f}%)")

    # Group by unique user prompt to guarantee 0% leakage
    prompt_to_samples = {}
    for s in valid_samples:
        prompt_to_samples.setdefault(s["user_prompt"], []).append(s)

    unique_prompts = list(prompt_to_samples.keys())
    random.seed(SEED)
    random.shuffle(unique_prompts)

    n_prompts = len(unique_prompts)
    n_train_p = int(n_prompts * 0.80)
    n_val_p = int(n_prompts * 0.10)

    train_prompts = set(unique_prompts[:n_train_p])
    val_prompts = set(unique_prompts[n_train_p:n_train_p + n_val_p])
    test_prompts = set(unique_prompts[n_train_p + n_val_p:])

    train_data = [s for p in train_prompts for s in prompt_to_samples[p]]
    val_data = [s for p in val_prompts for s in prompt_to_samples[p]]
    test_data = [s for p in test_prompts for s in prompt_to_samples[p]]

    print(f"\nDataset Splits:")
    print(f"  Training set   : {len(train_data):4d} samples ({len(train_data)/len(valid_samples)*100:.1f}%)")
    print(f"  Validation set : {len(val_data):4d} samples ({len(val_data)/len(valid_samples)*100:.1f}%)")
    print(f"  Test set       : {len(test_data):4d} samples ({len(test_data)/len(valid_samples)*100:.1f}%)")

    # Strict leakage assertion
    train_leak = train_prompts.intersection(val_prompts)
    test_leak = train_prompts.intersection(test_prompts)
    val_leak = val_prompts.intersection(test_prompts)
    assert len(train_leak) == 0, f"Train-Val leakage: {train_leak}"
    assert len(test_leak) == 0, f"Train-Test leakage: {test_leak}"
    assert len(val_leak) == 0, f"Val-Test leakage: {val_leak}"
    print("Zero prompt leakage confirmed across Train, Validation, and Test sets!")

    # Write files
    DATA_DIR_V4.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, samples):
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_DATA_PATH, train_data)
    write_jsonl(VAL_DATA_PATH, val_data)
    write_jsonl(TEST_DATA_PATH, test_data)

    report = {
        "dataset_name": "MiniGPT V4 Conversational Instruction Dataset",
        "version": "v4.0.0",
        "description": "Multi-category instruction-following conversational dataset tailored for MiniGPT V4 context window (128 tokens) with zero cross-split leakage.",
        "license": "MIT / Creative Commons Zero (CC0)",
        "origin": "Curated instruction pairs spanning 8 target categories with deterministic splits.",
        "random_seed": SEED,
        "context_window_limit": BLOCK_SIZE,
        "total_samples": len(valid_samples),
        "split_counts": {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data),
        },
        "category_distribution": category_counts,
        "token_statistics": {
            "min_tokens": min(s["token_count"] for s in valid_samples),
            "max_tokens": max(s["token_count"] for s in valid_samples),
            "avg_tokens": round(sum(s["token_count"] for s in valid_samples) / len(valid_samples), 2),
        },
    }

    with open(DATASET_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nSaved datasets and report to: {DATA_DIR_V4}")
    print("Zero-leakage V4 dataset generation complete!")


if __name__ == "__main__":
    main()
