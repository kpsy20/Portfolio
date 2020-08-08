using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class MenuScript : MonoBehaviour {
    public Text title, fake;
    public Button fakeb;
	void Update () {
        if(Application.platform == RuntimePlatform.Android)
        {
            if(Input.GetKey(KeyCode.Escape))
            {
                Application.Quit();
            }
        }
        if(ButtonController.lang == true)
        {
            title.text = "쩨비뽑기";
            if (InputController.enter == false) { fake.text = "주작하기"; }
            if(InputController.enter==true)
            {
                fake.text = "준비중";
                fakeb.enabled = false;
            }
            
        }
        else
        {
            title.text = "Drawing lots";
            if (InputController.enter == false) { fake.text = "Manipulation";  }
            if(InputController.enter == true)
            {
                fake.text = "Preparing";
                fakeb.enabled = false;
            }
           
        }
    }
}
