async function sendQuery() {
  const query = document.getElementById("query").value;
  const res = await fetch("http://localhost:8000/hackrx/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });
  const data = await res.json();
  document.getElementById("response").textContent = JSON.stringify(data, null, 2);
}
