const tryhard = () => {
    return new Promise((resolve,reject) => {
        setTimeout(() => {
            let success = true;
            if(success){
                resolve("True");
            }
            else{
                resolve("False");
            }
        })
    })
}; 

async function name(params){
    try{
        const result = await tryhard();
        console.log(result);
        return result;
    }
    catch{
        console.error("An error occurred");
        return "Error";
    }
}