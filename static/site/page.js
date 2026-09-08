(()=>{
const canvas=document.getElementById('character');
const character=new PixelCharacter(canvas,new URL('site/sprites.png',document.baseURI).href);
character.x=220;
character.loopScenes=true;

const sequence=[
  {state:'idle',duration:2400},
  {state:'walk',direction:1,duration:1600},
  {state:'blackboard',duration:5200},
  {state:'idle',duration:2400},
  {state:'walk',direction:-1,duration:1600},
  {state:'watering',duration:5200}
];
let step=0,timer=null,introTimer=null,paused=false,ready=false,stepDeadline=0,greeting=false;

function clear(){if(timer!==null)clearTimeout(timer);timer=null;}
function clearIntro(){if(introTimer!==null)clearTimeout(introTimer);introTimer=null;}
function playStep(){
  clear();
  if(paused)return;
  const current=sequence[step];
  character.moving=0;
  if(current.state==='walk')character.move(current.direction);
  else character.setState(current.state);
  stepDeadline=performance.now()+current.duration;
  timer=setTimeout(()=>{
    step=(step+1)%sequence.length;
    playStep();
  },current.duration);
}
function snapshot(){
  return {
    step,
    remaining:Math.max(0,stepDeadline-performance.now()),
    state:character.state,
    phase:character.phase,
    elapsed:character.elapsed,
    moving:character.moving,
    direction:character.direction,
    x:character.x,
    boardScene:character.boardScene&&{...character.boardScene}
  };
}
function resume(saved){
  step=saved.step;
  character.x=saved.x;
  character.direction=saved.direction;
  character.moving=0;
  character.setState(saved.state);
  character.x=saved.x;
  character.direction=saved.direction;
  character.moving=saved.moving;
  character.elapsed=saved.elapsed;
  character.phase=saved.phase;
  if(saved.boardScene)character.boardScene={...saved.boardScene};
  stepDeadline=performance.now()+saved.remaining;
  timer=setTimeout(()=>{
    step=(step+1)%sequence.length;
    playStep();
  },saved.remaining);
}
function greet(restart=false){
  if(paused||!ready||greeting||character.state==='wave')return;
  const saved=restart?null:snapshot();
  clear();clearIntro();
  greeting=true;
  character.moving=0;
  character.setState('wave');
  timer=setTimeout(()=>{
    greeting=false;
    if(saved)resume(saved);else{step=0;playStep();}
  },1900);
}
function setPaused(value){
  paused=value;
  character.pause(value);
  if(value){clear();clearIntro();}else playStep();
}

const reduced=matchMedia('(prefers-reduced-motion: reduce)');
reduced.addEventListener('change',event=>setPaused(event.matches));
document.addEventListener('visibilitychange',()=>setPaused(document.hidden||reduced.matches));
canvas.addEventListener('pointermove',event=>{
  if(event.pointerType==='touch'||!ready||paused)return;
  const rect=canvas.getBoundingClientRect();
  const x=(event.clientX-rect.left)*canvas.width/rect.width;
  const y=(event.clientY-rect.top)*canvas.height/rect.height;
  const inside=Math.abs(x-character.x)<85&&y>character.y-235&&y<character.y+12;
  if(inside&&!canvas.matches(':hover[data-greeting]')){
    canvas.dataset.greeting='true';
    greet();
  }
  canvas.style.cursor=inside?'pointer':'';
  if(!inside)delete canvas.dataset.greeting;
});
canvas.addEventListener('pointerleave',()=>{delete canvas.dataset.greeting;canvas.style.cursor='';});
window.addEventListener('pagehide',()=>{clear();clearIntro();character.destroy();});
character.ready.then(()=>{
  ready=true;
  if(reduced.matches){setPaused(true);return;}
  playStep();
  introTimer=setTimeout(()=>{introTimer=null;greet(true);},1400);
}).catch(()=>{
  canvas.setAttribute('aria-label','The character could not load. Please refresh.');
});
window.character=character;
})();
