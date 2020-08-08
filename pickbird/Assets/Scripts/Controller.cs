using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class Controller : MonoBehaviour {

    public DataController dataController;
    public Button button;
    public Text displayer;
	// Update is called once per frame
	void Update () {
        displayer.text = dataController.GetNum().ToString();
	}
    
    //광고 볼때
    public void Upgrade()
    {
        dataController.AddNum(1);
    }
}
