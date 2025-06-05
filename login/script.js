document.getElementById("loginForm").addEventListener("submit", function(e) {
  e.preventDefault(); // Prevent form submission

  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();
  const error = document.getElementById("error");

  // Simple validation
  if (username === "admin" && password === "1234") {
    alert("Login successful!");
    error.textContent = "";
    // Redirect or continue...
  } else {
    error.textContent = "Invalid username or password.";
  }
});
