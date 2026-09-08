(function(scope){
class ScrollDirector{
  constructor(character,options={}){
    this.character=character;this.section=options.section||(()=> 'idle');
    // Call browser timers as globals so their receiver stays the Window.
    this.schedule=options.schedule||((fn,delay)=>setTimeout(fn,delay));
    this.cancel=options.cancel||((id)=>clearTimeout(id));
    this.beforeInterrupt=options.beforeInterrupt||(()=>{});
    this.active=this.section();this.paused=false;this.timer=null;this.scrolling=false;
  }
  static sceneAt(blocks,height,center=height/2){
    const padding=Math.min(24,height*.04);
    return blocks.find(({rect})=>rect.top<=center+padding&&rect.bottom>=center-padding)?.scene||'idle';
  }
  clear(){if(this.timer!==null)this.cancel(this.timer);this.timer=null;}
  enter(){const next=this.section();if(next!==this.active){this.active=next;}}
  start(){this.enter();if(!this.paused)this.character.play('wave');}
  scroll(){
    this.enter();this.clear();if(this.paused)return;
    const c=this.character;
    const target=this.active==='idle'?'walk':this.active;
    if(c.state!==target){
      if(c.state==='blackboard'||c.state==='watering')this.beforeInterrupt();
      c.setState(target);
    }
    this.scrolling=true;c.moving=0;
    this.timer=this.schedule(()=>this.stop(),220);
  }
  greet(){
    if(this.paused||this.scrolling||['wave','blackboard','watering'].includes(this.character.state))return;
    this.clear();
    this.character.moving=0;this.character.setState('wave');
  }
  stop(){
    this.clear();this.enter();this.scrolling=false;if(this.paused)return;
    const c=this.character;c.moving=0;
    if(this.active!=='idle'&&c.state===this.active)return;
    c.setState('idle');
    if(this.active!=='idle'){
      const target=this.active;
      this.timer=this.schedule(()=>{
        this.timer=null;this.enter();
        if(this.paused||this.scrolling||target!==this.active)return;
        c.play(target);
      },120);
    }
  }
  setPaused(paused){this.paused=paused;this.clear();this.character.pause(paused);if(!paused)this.stop();}
  destroy(){this.clear();}
}
if(typeof module!=='undefined'&&module.exports)module.exports=ScrollDirector;else scope.ScrollDirector=ScrollDirector;
})(typeof window!=='undefined'?window:globalThis);
