import os
import random

import numpy as np
import streamlit as st
import matplotlib as plt
from openai import OpenAI

# --------- Perplexity / Sonar client ---------
# Requires PERPLEXITY_API_KEY in your environment
client = OpenAI(
    api_key=os.environ.get("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai",
)

MODEL_NAME = "sonar-pro"

# --------- GA config (some via UI) ---------
GRID_SIZE = 8
POP_SIZE = 36
ELITE_COUNT = 6


# --------- GA helpers ---------
def make_target_pattern():
    """8x8 'pixel person' silhouette: head, body, arms, legs."""
    p = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # head (row 1, columns 3–4)
    p[1, 3:5] = 1

    # body (rows 2–4, column 4)
    p[2:5, 4] = 1

    # arms (row 3, columns 2–6)
    p[3, 2:7] = 1

    # legs (rows 5–6, columns 3 and 5)
    p[5:7, 3] = 1
    p[5:7, 5] = 1

    return p


TARGET = make_target_pattern()


def random_creature():
    return np.random.randint(0, 2, size=(GRID_SIZE, GRID_SIZE), dtype=int)


def fitness(creature):
    return int(np.sum(creature == TARGET))


def mutate(creature, mutation_rate: float):
    child = creature.copy()
    mask = np.random.rand(GRID_SIZE, GRID_SIZE) < mutation_rate
    child[mask] = 1 - child[mask]
    return child


def init_population():
    return [random_creature() for _ in range(POP_SIZE)]


def step_generation(population, mutation_rate: float):
    scored = [(fitness(c), c) for c in population]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_fit, best_creature = scored[0]

    elites = [c for _, c in scored[:ELITE_COUNT]]
    new_population = []
    while len(new_population) < POP_SIZE:
        parent = random.choice(elites)
        child = mutate(parent, mutation_rate)
        new_population.append(child)

    avg_fit = sum(f for f, _ in scored) / len(scored)
    return new_population, best_creature, best_fit, avg_fit


# --------- Visualization helpers ---------
def plot_population_grid(population, best_creature, best_fit, generation):
    rows = cols = int(np.sqrt(POP_SIZE))
    fig, axes = plt.subplots(rows, cols + 1, figsize=(10, 6))

    # Target
    axes[0, 0].imshow(TARGET, cmap="Greys", vmin=0, vmax=1)
    axes[0, 0].set_title("Target\n(pixel person)", fontsize=8)
    axes[0, 0].axis("off")

    # Best creature
    axes[1, 0].imshow(best_creature, cmap="Greys", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Best (fit={best_fit})", fontsize=8)
    axes[1, 0].axis("off")

    for r in range(2, rows):
        axes[r, 0].axis("off")

    idx = 0
    for r in range(rows):
        for c in range(1, cols + 1):
            if idx < len(population):
                axes[r, c].imshow(population[idx], cmap="Greys", vmin=0, vmax=1)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
                idx += 1
            else:
                axes[r, c].axis("off")

    fig.suptitle(f"Generation {generation} (best fitness {best_fit})", fontsize=10)
    plt.tight_layout()
    return fig


# --------- Sonar explanation helper ---------
def explain_with_sonar(generation, best_fit, avg_fit, mutation_rate):
    max_fit = GRID_SIZE * GRID_SIZE
    user_text = f"""
We are running a tiny evolutionary algorithm on 8x8 pixel creatures.

Details of the current state:
- Target shape is a small pixel 'person' silhouette.
- Generation: {generation}
- Best fitness this generation: {best_fit} (out of {max_fit})
- Average fitness this generation: {avg_fit:.1f}
- Mutation rate: {mutation_rate:.3f} per pixel

Please explain, in simple language for first-year college students:
1) What this generation number and fitness information tell us about how far evolution has progressed.
2) What the current mutation rate likely does to the population (too wild, too tame, or reasonable).
3) One short suggestion for a change the student could try (e.g., mutation rate up/down, more generations) and what they should watch for.

Keep it under 5 short sentences.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly, concise tutor explaining evolutionary algorithms "
                "for an introductory science class. Use plain language, no jargon."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=250,
    )

    return completion.choices[0].message.content.strip()


# --------- Streamlit app ---------
st.set_page_config(page_title="Pixel Evolution Lab", page_icon="🧬", layout="wide")

st.title("Pixel Evolution Lab")
st.caption(
    "Evolving an 8×8 pixel person using a simple genetic algorithm, "
    "with Sonar-pro explaining what is happening."
)

# Sidebar controls
st.sidebar.header("Evolution controls")
mutation_rate = st.sidebar.slider(
    "Mutation rate (probability each pixel flips)",
    min_value=0.0,
    max_value=0.2,
    value=0.02,
    step=0.005,
)
generations_to_run = st.sidebar.selectbox(
    "Run how many generations at once?",
    [1, 5, 10, 25],
    index=0,
)

# Initialize state
if "population" not in st.session_state:
    st.session_state.population = init_population()
    st.session_state.generation = 0
    st.session_state.best_fit = 0
    st.session_state.avg_fit = 0.0
    st.session_state.best_creature = st.session_state.population[0]
    st.session_state.chat_history = []

# Layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Evolution playground")

    col_buttons = st.columns(3)
    with col_buttons[0]:
        if st.button("Initialize / Reset"):
            st.session_state.population = init_population()
            st.session_state.generation = 0
            st.session_state.best_fit = 0
            st.session_state.avg_fit = 0.0
            st.session_state.best_creature = st.session_state.population[0]

    with col_buttons[1]:
        if st.button(f"Run {generations_to_run} generation(s)"):
            pop = st.session_state.population
            best, best_fit, avg_fit = None, None, None
            for _ in range(generations_to_run):
                pop, best, best_fit, avg_fit = step_generation(pop, mutation_rate)
                st.session_state.generation += 1
            st.session_state.population = pop
            st.session_state.best_creature = best
            st.session_state.best_fit = best_fit
            st.session_state.avg_fit = avg_fit

    with col_buttons[2]:
        if st.button("Explain this with AI"):
            try:
                explanation = explain_with_sonar(
                    st.session_state.generation,
                    st.session_state.best_fit,
                    st.session_state.avg_fit,
                    mutation_rate,
                )
            except Exception as e:
                explanation = f"(Error talking to Sonar: {e})"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": explanation}
            )

    # Show current stats and grid
    scored_now = [(fitness(c), c) for c in st.session_state.population]
    scored_now.sort(key=lambda x: x[0], reverse=True)
    best_fit_now, best_creature_now = scored_now[0]

    # sync best if needed
    st.session_state.best_creature = best_creature_now
    st.session_state.best_fit = best_fit_now

    st.markdown(f"**Generation:** {st.session_state.generation}")
    st.markdown(
        f"**Best fitness:** {best_fit_now} / {GRID_SIZE * GRID_SIZE}  "
        f"(average ≈ {st.session_state.avg_fit:.1f})"
    )
    st.markdown(f"**Mutation rate:** {mutation_rate:.3f}")

    fig = plot_population_grid(
        st.session_state.population,
        st.session_state.best_creature,
        st.session_state.best_fit,
        st.session_state.generation,
    )
    st.pyplot(fig)

with right:
    st.subheader("Ask the tutor (Sonar‑pro)")

    # Show automatic explanations that have been generated
    for msg in st.session_state.chat_history:
        st.markdown(msg["content"])
        st.markdown("---")

    user_q = st.text_input(
        "Ask a question about what you see",
        placeholder="e.g., Why did progress slow down after a while?",
    )

    if st.button("Ask AI about this run"):
        if user_q.strip():
            try:
                gen = st.session_state.generation
                best_fit = st.session_state.best_fit
                avg_fit = st.session_state.avg_fit
                max_fit = GRID_SIZE * GRID_SIZE

                prompt = f"""
We are in generation {gen} of an 8x8 pixel-person evolution.

Stats:
- Best fitness: {best_fit} / {max_fit}
- Average fitness: {avg_fit:.1f} / {max_fit}
- Mutation rate: {mutation_rate:.3f}

The student asks:
{user_q}

Explain in 3–6 short sentences, in very plain language.
Reference these numbers explicitly so they can connect their question to what is happening.
"""
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a patient tutor explaining evolutionary algorithms "
                            "and genetic algorithms to introductory college students."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=300,
                )
                answer = completion.choices[0].message.content.strip()
            except Exception as e:
                answer = f"(Error talking to Sonar: {e})"

            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"**Q:** {user_q}\n\n{answer}"}
            )
            st.experimental_rerun()