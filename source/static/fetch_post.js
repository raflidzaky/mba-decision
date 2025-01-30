// POST request script to return requested output

fetch('http://127.0.0.1:8000/predict', {
    method: 'POST', // Specify the method as POST
    headers: {
        'Content-Type': 'application/json', // Set the Content-Type to JSON
    },
    body: JSON.stringify({
        "features": [9.9, 9.9]
         // Replace this with the value you want to send
    }),
})
.then(response => response.json())  // Parse the response as JSON
.then(data => console.log(data))     // Log the response to the console
.catch(error => console.error('Error:', error));  // Handle any errors