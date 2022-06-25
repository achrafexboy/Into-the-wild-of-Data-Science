const box = document.querySelector('.box');
const navBar = document.querySelector('.navBar');
const navBarElemnts = document.querySelectorAll('.navBarElemnt');
const navBarContainer = document.querySelector('.navBarContainer');
const theBody = document.querySelector('.theBody');


//nav bar box
box.addEventListener('click', e => {
    e.target.classList.toggle('active');
    navBar.classList.toggle('active');
    navBarContainer.classList.toggle('active');
    theBody.classList.toggle('active');
})

navBarElemnts.forEach((elem) => {
    elem.addEventListener('click', () => {
        box.classList.toggle('active');
        navBar.classList.toggle('active');
        navBarContainer.classList.toggle('active');
        body.classList.toggle('active');
    }) 
})