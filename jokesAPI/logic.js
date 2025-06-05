rnd = document.querySelector("#rndJokes");
src = document.querySelector("#SearchJokes");
cnt = document.querySelector("#container");
input = document.querySelector("#searchInput")

let clearContent = () => {
  let cnt = document.querySelector("#container").innerHTML = "";
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

async function getRndJoke(){
    clearContent();
    const response = await fetch("https://icanhazdadjoke.com/",{
        headers: {
            "Accept": "application/json"
        }
    });
    const data = await response.json();
    cnt.innerText = data.joke;
    console.log(data);
}

async function getSearchJoke(){
    clearContent();
    const term = input.value;

    if(!term){
        alert("Must provide domain");
        return;
    }

    const response = await fetch(`https://icanhazdadjoke.com/search?term=${term}`, {
        headers: {
            "Accept": "application/json"
        }
    });

    const data = await response.json();

    if(data.results.length === 0) {
        cnt.innerText = "No jokes found";
        return;
    }

    shuffleArray(data.results);
    const selectedJokes = data.results.slice(0, 5);

    const ul = document.createElement("ul");
    selectedJokes.forEach(jokeObj => {
        const li = document.createElement("li");
        li.innerText = jokeObj.joke;
        ul.appendChild(li);
    });

    cnt.appendChild(ul);
}

if(rnd) {
    getRndJoke();
}
else if(src) {
    getSearchJoke();
}
/*rnd.addEventListener("click", getRndJoke);
src.addEventListener("click", getSearchJoke);*/