(()=>{
const themeButton=document.getElementById('theme-toggle');
function setTheme(dark){
  document.documentElement.dataset.theme=dark?'dark':'light';
  themeButton.setAttribute('aria-pressed',String(dark));
  themeButton.setAttribute('aria-label',dark?'Switch to light mode':'Switch to dark mode');
  document.querySelector('meta[name="theme-color"]').content=dark?'#18191b':'#ffffff';
}
setTheme(document.documentElement.dataset.theme==='dark');
themeButton.addEventListener('click',()=>{
  const dark=document.documentElement.dataset.theme!=='dark';setTheme(dark);
  try{localStorage.setItem('color-theme',dark?'dark':'light');}catch{}
});
})();
